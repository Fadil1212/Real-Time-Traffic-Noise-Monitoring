#!/usr/bin/env python3
from flask import Flask, request, jsonify
import psycopg2
import logging
from datetime import datetime, timedelta
import os
import heapq
from collections import defaultdict
import math
import threading
import time

from road_network import road_network

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'traffic_noise_db'),
    'user': os.getenv('DB_USER', 'traffic_user'),
    'password': os.getenv('DB_PASSWORD', 'traffic_pass')
}


class GraphCache:
    """
    CACHED GRAPH
    Build graph once, reuse for all requests
    Update every 5 minutes in background
    """
    def __init__(self):
        self.graph = None
        self.locations = None
        self.last_updated = None
        self.is_building = False
        self.update_interval = 300  # 5 minutes
        self.lock = threading.Lock()
    
    def is_stale(self):
        """Check if cache needs refresh"""
        if not self.graph or not self.last_updated:
            return True
        age = (datetime.now() - self.last_updated).total_seconds()
        return age > self.update_interval
    
    def get(self):
        """Get cached graph (thread-safe)"""
        with self.lock:
            return self.graph, self.locations, self.last_updated
    
    def set(self, graph, locations):
        """Update cache (thread-safe)"""
        with self.lock:
            self.graph = graph
            self.locations = locations
            self.last_updated = datetime.now()
            self.is_building = False
            logger.info(f"✅ Graph cached: {len(locations)} locations, {sum(len(e) for e in graph.values())//2} edges")


class RouteRecommender:
    """Optimized route calculator with caching"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.road_network_initialized = False
        self.cache = GraphCache()
        
        # Start background graph updater
        self.start_background_updater()
    
    def start_background_updater(self):
        """Start thread to update graph every 5 minutes"""
        def updater():
            while True:
                try:
                    if self.cache.is_stale() and not self.cache.is_building:
                        logger.info("🔄 Background: Refreshing graph cache...")
                        self.cache.is_building = True
                        self.refresh_graph()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"❌ Background updater error: {e}")
                    self.cache.is_building = False
                    time.sleep(60)
        
        thread = threading.Thread(target=updater, daemon=True)
        thread.start()
        logger.info("✅ Background graph updater started")
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            if self.conn and not self.conn.closed:
                return True
            self.conn = psycopg2.connect(**self.db_config)
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def initialize_road_network(self):
        """Initialize Milwaukee road network"""
        if not self.road_network_initialized:
            logger.info("🗺️  Initializing Milwaukee road network...")
            if road_network.initialize():
                self.road_network_initialized = True
                logger.info("✅ Road network initialized")
            else:
                logger.warning("⚠️  Road network initialization failed, using fallback coordinates")
        return self.road_network_initialized
    
    def get_route_segments(self):
        """Get all route segments"""
        query = """
        SELECT DISTINCT
            street_name,
            neighborhood,
            center_lat,
            center_lon,
            avg_noise_level,
            noise_category
        FROM route_noise_segments
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        ORDER BY street_name
        """
        
        cursor = self.conn.cursor()
        cursor.execute(query)
        segments = cursor.fetchall()
        cursor.close()
        
        logger.info(f"📊 Fetched {len(segments)} route segments")
        return segments
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using haversine formula (meters)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def build_hierarchical_graph(self, segments):
        """
        Build graph with optimizations
        - Spatial indexing for faster neighbor queries
        - Early termination when enough connections found
        - Reduced redundant calculations
        """
        graph = defaultdict(list)
        locations = {}
        
        # Store all locations
        for segment in segments:
            street_name, neighborhood, lat, lon, noise, category = segment
            location_key = f"{street_name}|{neighborhood}"
            locations[location_key] = {
                'street': street_name,
                'neighborhood': neighborhood,
                'lat': lat,
                'lon': lon,
                'noise': float(noise) if noise else 70.0,
                'category': category
            }
        
        location_list = list(locations.keys())
        total_locations = len(location_list)
        logger.info(f"🏙️  Building optimized graph for {total_locations} locations...")
        
        # ========================================
        # OPTIMIZATION: Spatial bucketing
        # Group locations by lat/lon grid to reduce comparisons
        # ========================================
        grid_size = 0.01  # ~1km grid cells
        spatial_buckets = defaultdict(list)
        
        for loc_key in location_list:
            loc = locations[loc_key]
            bucket_key = (round(loc['lat'] / grid_size), round(loc['lon'] / grid_size))
            spatial_buckets[bucket_key].append(loc_key)
        
        logger.info(f"  📐 Spatial index: {len(spatial_buckets)} buckets")
        
        # ========================================
        # TIER 1: LOCAL CONNECTIONS (< 3km)
        # Only check nearby buckets (9 cells max)
        # ========================================
        tier1_count = 0
        for loc_key in location_list:
            loc = locations[loc_key]
            my_bucket = (round(loc['lat'] / grid_size), round(loc['lon'] / grid_size))
            
            # Check only neighboring buckets (3x3 grid around current location)
            nearby_buckets = [
                (my_bucket[0] + dx, my_bucket[1] + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
            ]
            
            nearby_locations = []
            for bucket in nearby_buckets:
                nearby_locations.extend(spatial_buckets.get(bucket, []))
            
            # Connect to nearby locations
            for other_key in nearby_locations:
                if other_key <= loc_key:  # Avoid duplicates
                    continue
                
                other = locations[other_key]
                distance = self.calculate_distance(
                    loc['lat'], loc['lon'],
                    other['lat'], other['lon']
                )
                
                if distance < 3000:  # Local: < 3km
                    noise_weight = (loc['noise'] + other['noise']) / 2
                    weight = distance + (noise_weight * 10)
                    
                    graph[loc_key].append((other_key, weight, distance, noise_weight, 'local'))
                    graph[other_key].append((loc_key, weight, distance, noise_weight, 'local'))
                    tier1_count += 1
        
        logger.info(f"  ✅ Tier 1 (Local < 3km): {tier1_count} connections")
        
        # ========================================
        # TIER 2: REGIONAL CONNECTIONS (3-10km)
        # OPTIMIZATION: Only for under-connected nodes
        # ========================================
        tier2_count = 0
        for loc_key in location_list:
            # Skip if already well-connected
            if len(graph[loc_key]) >= 5:
                continue
            
            loc = locations[loc_key]
            
            # Find closest 3 locations in 3-10km range
            candidates = []
            for other_key in location_list:
                if other_key == loc_key:
                    continue
                
                other = locations[other_key]
                distance = self.calculate_distance(
                    loc['lat'], loc['lon'],
                    other['lat'], other['lon']
                )
                
                if 3000 <= distance < 10000:
                    candidates.append((distance, other_key))
            
            # Connect to 3 closest
            candidates.sort()
            for distance, other_key in candidates[:3]:
                # Check if not already connected
                if not any(e[0] == other_key for e in graph[loc_key]):
                    other = locations[other_key]
                    noise_weight = (loc['noise'] + other['noise']) / 2
                    weight = distance * 1.2 + (noise_weight * 15)
                    
                    graph[loc_key].append((other_key, weight, distance, noise_weight, 'regional'))
                    graph[other_key].append((loc_key, weight, distance, noise_weight, 'regional'))
                    tier2_count += 1
        
        logger.info(f"  ✅ Tier 2 (Regional 3-10km): {tier2_count} connections")
        
        # ========================================
        # TIER 3: CITY-WIDE CONNECTIONS (10-30km)
        # OPTIMIZATION: Only for isolated nodes
        # ========================================
        tier3_count = 0
        for loc_key in location_list:
            # Skip if already connected
            if len(graph[loc_key]) >= 3:
                continue
            
            loc = locations[loc_key]
            
            # Find 2 closest locations regardless of distance
            candidates = []
            for other_key in location_list:
                if other_key == loc_key:
                    continue
                
                other = locations[other_key]
                distance = self.calculate_distance(
                    loc['lat'], loc['lon'],
                    other['lat'], other['lon']
                )
                
                candidates.append((distance, other_key))
            
            # Connect to 2 closest
            candidates.sort()
            for distance, other_key in candidates[:2]:
                if not any(e[0] == other_key for e in graph[loc_key]):
                    other = locations[other_key]
                    noise_weight = (loc['noise'] + other['noise']) / 2
                    weight = distance * 1.5 + (noise_weight * 20)
                    
                    graph[loc_key].append((other_key, weight, distance, noise_weight, 'citywide'))
                    graph[other_key].append((loc_key, weight, distance, noise_weight, 'citywide'))
                    tier3_count += 1
        
        logger.info(f"  ✅ Tier 3 (City-wide 10-30km): {tier3_count} connections")
        
        # Summary
        total_edges = sum(len(edges) for edges in graph.values()) // 2
        avg_connections = sum(len(edges) for edges in graph.values()) / len(graph) if graph else 0
        
        logger.info(f"🗺️  OPTIMIZED GRAPH COMPLETE:")
        logger.info(f"     Nodes: {len(locations)}")
        logger.info(f"     Total Edges: {total_edges}")
        logger.info(f"     Avg Connections/Node: {avg_connections:.1f}")
        
        return graph, locations
    
    def refresh_graph(self):
        """Refresh the cached graph"""
        try:
            self.connect()
            segments = self.get_route_segments()
            
            if segments:
                graph, locations = self.build_hierarchical_graph(segments)
                self.cache.set(graph, locations)
                return True
            else:
                logger.warning("⚠️  No segments to build graph")
                return False
        except Exception as e:
            logger.error(f"❌ Graph refresh failed: {e}")
            self.cache.is_building = False
            return False
    
    def get_cached_graph(self):
        """
        FIX #1: Get graph from cache or build if needed
        """
        graph, locations, last_updated = self.cache.get()
        
        if self.cache.is_stale():
            if not self.cache.is_building:
                logger.info("🔄 Cache stale, rebuilding...")
                self.cache.is_building = True
                
                # Try to rebuild
                if self.refresh_graph():
                    graph, locations, last_updated = self.cache.get()
                else:
                    self.cache.is_building = False
        
        if not graph:
            logger.info("🔄 No cache, building initial graph...")
            if self.refresh_graph():
                graph, locations, last_updated = self.cache.get()
        
        if graph:
            age = (datetime.now() - last_updated).total_seconds() if last_updated else 0
            logger.info(f"✅ Using cached graph (age: {age:.0f}s, {len(locations)} locations)")
        
        return graph, locations
    
    def dijkstra(self, graph, start, end, weight_type='balanced'):
        """
        PTIMIZED DIJKSTRA
        - Reduced max iterations
        - Early termination
        - Better heuristic
        """
        if start not in graph:
            logger.error(f"❌ Start node not in graph")
            return None, None, None
        
        if end not in graph:
            logger.error(f"❌ End node not in graph")
            return None, None, None
        
        queue = [(0, start, [start], 0, 0)]
        visited = set()
        iterations = 0
        
        # FIX #2: Reduced from len(graph) * 200 to len(graph) * 50
        # For 200 nodes: 40,000 → 10,000 iterations max
        max_iterations = len(graph) * 50
        
        while queue and iterations < max_iterations:
            iterations += 1
            cost, node, path, total_dist, total_noise = heapq.heappop(queue)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node == end:
                avg_noise = total_noise / len(path) if len(path) > 0 else 0
                logger.info(f"✅ Found path in {iterations} iterations: {len(path)} hops, {total_dist/1000:.2f}km, {avg_noise:.1f}dB")
                return path, total_dist, avg_noise
            
            # FIX #2: Early termination if path getting too long
            if len(path) > 50:  # Milwaukee shouldn't need >50 hops
                continue
            
            for neighbor_info in graph[node]:
                neighbor = neighbor_info[0]
                edge_weight = neighbor_info[1]
                distance = neighbor_info[2]
                noise = neighbor_info[3]
                
                if neighbor not in visited:
                    if weight_type == 'quietest':
                        new_cost = cost + (noise * 20)
                    elif weight_type == 'shortest':
                        new_cost = cost + distance
                    else:  # balanced
                        new_cost = cost + distance + (noise * 10)
                    
                    new_path = path + [neighbor]
                    new_dist = total_dist + distance
                    new_noise = total_noise + noise
                    
                    heapq.heappush(queue, (new_cost, neighbor, new_path, new_dist, new_noise))
        
        logger.warning(f"❌ No path found after {iterations} iterations")
        return None, None, None
    
    def get_simplified_coordinates(self, waypoint_locations):
        """
        SIMPLIFIED COORDINATE GENERATION
        
        Instead of making OSMnx API call for EVERY segment:
        - Use straight lines between waypoints
        - Add intermediate points for smooth curves
        - Optional: Only use OSMnx for critical segments
        """
        all_coords = []
        
        for i in range(len(waypoint_locations) - 1):
            loc1 = waypoint_locations[i]
            loc2 = waypoint_locations[i + 1]
            
            # Check distance - only use OSMnx for short segments
            distance = self.calculate_distance(
                loc1['lat'], loc1['lon'],
                loc2['lat'], loc2['lon']
            )
            
            # Use road network for short segments only (<2km)
            if distance < 2000 and self.road_network_initialized:
                route_coords = road_network.get_route_between_sensors(
                    loc1['lat'], loc1['lon'],
                    loc2['lat'], loc2['lon']
                )
                
                if route_coords and len(route_coords) > 2:
                    for lat, lon in route_coords:
                        all_coords.append([float(lon), float(lat)])
                else:
                    # Fallback: straight line with intermediate points
                    all_coords.extend(self._interpolate_line(
                        loc1['lon'], loc1['lat'],
                        loc2['lon'], loc2['lat'],
                        num_points=5
                    ))
            else:
                # FIX #3: For long distances, just use interpolated straight line
                all_coords.extend(self._interpolate_line(
                    loc1['lon'], loc1['lat'],
                    loc2['lon'], loc2['lat'],
                    num_points=max(3, int(distance / 1000))  # 1 point per km
                ))
        
        # Add final point
        if waypoint_locations:
            final = waypoint_locations[-1]
            all_coords.append([float(final['lon']), float(final['lat'])])
        
        logger.info(f"✅ Generated {len(all_coords)} coordinates (simplified)")
        return all_coords
    
    def _interpolate_line(self, lon1, lat1, lon2, lat2, num_points=5):
        """Create smooth line between two points"""
        coords = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            lon = lon1 + (lon2 - lon1) * t
            lat = lat1 + (lat2 - lat1) * t
            coords.append([float(lon), float(lat)])
        return coords
    
    def find_alternative_routes(self, origin, destination):
        """
        FIX #1-5: Optimized route finding with caching
        """
        try:
            # FIX #1: Use cached graph
            graph, locations = self.get_cached_graph()
            
            if not graph or not locations:
                return {
                    'success': False,
                    'error': 'No route data available',
                    'user_message': '🔄 System collecting data. Try again in a few minutes.'
                }
            
            if len(graph) < 2:
                return {
                    'success': False,
                    'error': f'Insufficient locations ({len(graph)})',
                    'user_message': f'⏳ Need more data. Currently tracking {len(graph)} locations.'
                }
            
            # Find origin and destination
            origin_key = None
            dest_key = None
            
            logger.info(f"🔍 Searching: {origin} → {destination}")
            
            for key, loc in locations.items():
                if origin.lower() in loc['street'].lower() or origin.lower() in loc['neighborhood'].lower():
                    if not origin_key:
                        origin_key = key
                        logger.info(f"✅ Origin: {loc['street']} ({loc['neighborhood']})")
                
                if destination.lower() in loc['street'].lower() or destination.lower() in loc['neighborhood'].lower():
                    if not dest_key:
                        dest_key = key
                        logger.info(f"✅ Destination: {loc['street']} ({loc['neighborhood']})")
            
            if not origin_key or not dest_key:
                available = [f"{loc['street']} ({loc['neighborhood']})" for loc in list(locations.values())[:15]]
                
                if not origin_key and not dest_key:
                    msg = f'❌ Could not find "{origin}" or "{destination}"'
                elif not origin_key:
                    msg = f'❌ Could not find origin "{origin}"'
                else:
                    msg = f'❌ Could not find destination "{destination}"'
                
                return {
                    'success': False,
                    'error': msg,
                    'user_message': msg,
                    'available_locations': available,
                    'suggestion': 'Use actual Milwaukee street names with noise sensors.'
                }
            
            # Calculate distance
            origin_loc = locations[origin_key]
            dest_loc = locations[dest_key]
            straight_dist = self.calculate_distance(
                origin_loc['lat'], origin_loc['lon'],
                dest_loc['lat'], dest_loc['lon']
            )
            
            logger.info(f"📏 Straight-line distance: {straight_dist/1000:.2f}km")
            
            # FIX #2: Calculate routes with optimized Dijkstra
            routes = []
            route_types = [
                ('quietest', 'Quietest'),
                ('shortest', 'Shortest'),
                ('balanced', 'Balanced')
            ]
            
            for route_type, route_name in route_types:
                logger.info(f"🔍 Calculating {route_type} route...")
                path, dist, noise = self.dijkstra(graph, origin_key, dest_key, route_type)
                
                if path and path not in [r.get('_raw_path') for r in routes]:
                    waypoint_locs = [locations[p] for p in path]
                    
                    # FIX #3: Use simplified coordinate generation
                    logger.info(f"📍 Generating coordinates for {len(waypoint_locs)} waypoints...")
                    road_coords = self.get_simplified_coordinates(waypoint_locs)
                    
                    routes.append({
                        'type': route_type,
                        'path': [locations[p]['street'] for p in path],
                        'path_coordinates': road_coords,
                        'neighborhoods': [locations[p]['neighborhood'] for p in path],
                        'distance_meters': round(dist, 0),
                        'avg_noise_level': round(noise, 1),
                        'estimated_time_minutes': round(dist / 83.33, 1),
                        'noise_category': self._get_noise_category(noise),
                        '_raw_path': path
                    })
            
            if not routes:
                return {
                    'success': False,
                    'error': 'No route found',
                    'user_message': f'🚫 No route between "{origin}" and "{destination}"',
                    'details': {
                        'straight_line_distance': f'{straight_dist/1000:.2f}km',
                        'graph_coverage': f'{len(graph)} locations'
                    },
                    'suggestion': f'Distance is {straight_dist/1000:.1f}km. Try locations closer together or wait for more data collection.',
                    'troubleshooting': [
                        'Ensure both locations are in Milwaukee',
                        'Try major streets or landmarks',
                        'Wait 5-10 minutes for more sensors',
                        f'Current system range: up to 30km'
                    ]
                }
            
            # Clean and rank
            for route in routes:
                route.pop('_raw_path', None)
            
            routes.sort(key=lambda x: x['avg_noise_level'])
            
            for i, route in enumerate(routes):
                route['rank'] = i + 1
                route['recommendation'] = self._get_recommendation(route)
            
            logger.info(f"✅ Generated {len(routes)} routes in total")
            
            self._store_recommendation(origin, destination, routes)
            
            return {
                'success': True,
                'origin': origin,
                'destination': destination,
                'routes': routes,
                'straight_line_distance_km': round(straight_dist/1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Route error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'user_message': '⚠️ Unexpected error. Try again.'
            }
    
    def _get_noise_category(self, noise_level):
        if noise_level >= 90:
            return 'critical'
        elif noise_level >= 80:
            return 'high'
        elif noise_level >= 70:
            return 'moderate'
        else:
            return 'low'
    
    def _get_recommendation(self, route):
        noise = route['avg_noise_level']
        dist_km = route['distance_meters'] / 1000
        
        if noise < 70:
            return f"✅ Quiet route ({dist_km:.1f}km) - Recommended"
        elif noise < 80:
            return f"⚠️ Moderate noise ({dist_km:.1f}km) - Acceptable"
        elif noise < 90:
            return f"⚠️ High noise ({dist_km:.1f}km) - Consider alternatives"
        else:
            return f"❌ Very loud ({dist_km:.1f}km) - Not recommended"
    
    def _store_recommendation(self, origin, destination, routes):
        """Store in database"""
        if not routes:
            return
        
        try:
            best_route = routes[0]
            cursor = self.conn.cursor()
            
            import json
            route_points = json.dumps({
                'path': best_route['path'],
                'neighborhoods': best_route['neighborhoods'],
                'coordinates': best_route['path_coordinates']
            })
            
            query = """
            INSERT INTO alternative_routes (
                route_name, origin_street, destination_street, route_points,
                avg_noise_level, estimated_duration_minutes, 
                noise_reduction_benefit, is_recommended
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """
            
            cursor.execute(query, (
                f"{origin} to {destination}",
                origin, destination, route_points,
                best_route['avg_noise_level'],
                best_route['estimated_time_minutes'],
                0.0,
                best_route['noise_category'] in ['low', 'moderate']
            ))
            
            route_id = cursor.fetchone()[0]
            
            query2 = """
            INSERT INTO route_recommendations (
                timestamp, user_origin, user_destination, recommended_route_id,
                current_noise_level, route_noise_level, reason
            ) VALUES (NOW(), %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query2, (
                origin, destination, route_id,
                best_route['avg_noise_level'],
                best_route['avg_noise_level'],
                best_route['recommendation']
            ))
            
            self.conn.commit()
            cursor.close()
            logger.info(f"✅ Stored recommendation")
            
        except Exception as e:
            logger.error(f"❌ Store failed: {e}")
            self.conn.rollback()


recommender = RouteRecommender(DB_CONFIG)

@app.before_request
def before_request():
    if not recommender.conn or recommender.conn.closed:
        recommender.connect()

@app.route('/health', methods=['GET'])
def health_check():
    graph, locations, last_updated = recommender.cache.get()
    cache_age = (datetime.now() - last_updated).total_seconds() if last_updated else None
    
    return jsonify({
        'status': 'healthy',
        'service': 'alternative-routes-optimized',
        'coverage': 'Milwaukee metro area (up to 30km)',
        'cache': {
            'cached': graph is not None,
            'locations': len(locations) if locations else 0,
            'age_seconds': round(cache_age) if cache_age else None,
            'is_building': recommender.cache.is_building
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/route-recommendations', methods=['POST'])
def get_route_recommendations():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided',
                'user_message': '❌ Provide origin and destination'
            }), 400
        
        origin = data.get('origin')
        destination = data.get('destination')
        
        if not origin or not destination:
            return jsonify({
                'success': False,
                'error': 'Missing origin or destination',
                'user_message': '❌ Specify both locations'
            }), 400
        
        logger.info(f"🗺️  OPTIMIZED REQUEST: {origin} → {destination}")
        
        result = recommender.find_alternative_routes(origin, destination)
        return jsonify(result), 200
            
    except Exception as e:
        logger.error(f"❌ API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'user_message': '⚠️ Server error'
        }), 500

@app.route('/api/locations', methods=['GET'])
def get_locations():
    try:
        # Get from cache if available
        graph, locations, last_updated = recommender.cache.get()
        
        if locations:
            location_list = [
                {'street': loc['street'], 'neighborhood': loc['neighborhood']}
                for loc in locations.values()
            ]
            
            return jsonify({
                'success': True,
                'count': len(location_list),
                'locations': sorted(location_list, key=lambda x: x['street'])[:100],
                'cached': True
            }), 200
        else:
            # Fallback to database
            query = """
            SELECT DISTINCT street_name, neighborhood
            FROM route_noise_segments
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY street_name, neighborhood
            LIMIT 100
            """
            
            cursor = recommender.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            locations = [{'street': r[0], 'neighborhood': r[1]} for r in results]
            
            return jsonify({
                'success': True,
                'count': len(locations),
                'locations': locations,
                'cached': False
            }), 200
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cache/refresh', methods=['POST'])
def refresh_cache():
    """Manual cache refresh endpoint"""
    try:
        logger.info("🔄 Manual cache refresh requested")
        if recommender.refresh_graph():
            graph, locations, last_updated = recommender.cache.get()
            return jsonify({
                'success': True,
                'message': 'Cache refreshed',
                'locations': len(locations) if locations else 0,
                'timestamp': last_updated.isoformat() if last_updated else None
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Cache refresh failed'
            }), 500
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🚀 Alternative Routes API - OPTIMIZED VERSION")
    logger.info("=" * 70)
    logger.info(f"📊 Database: {DB_CONFIG['host']}")
    logger.info(f"🌐 API: http://0.0.0.0:5000")
    logger.info(f"🗺️  Coverage: Entire Milwaukee (up to 30km routes)")
    logger.info(f"⚡ OPTIMIZATIONS:")
    logger.info(f"   ✅ Cached graph (5min refresh)")
    logger.info(f"   ✅ Spatial indexing")
    logger.info(f"   ✅ Reduced Dijkstra iterations (50x per node)")
    logger.info(f"   ✅ Simplified coordinates")
    logger.info(f"   ✅ Background updates")
    logger.info("=" * 70)
    
    if recommender.connect():
        logger.info("✅ Database connected")
        
        # Initialize road network
        recommender.initialize_road_network()
        
        # Build initial cache
        logger.info("🔄 Building initial graph cache...")
        if recommender.refresh_graph():
            logger.info("✅ Initial cache ready")
        else:
            logger.warning("⚠️  Initial cache build failed, will retry in background")
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("❌ Failed to connect")