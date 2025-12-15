"""
Milwaukee Road Network Manager
Handles road routing for heatmap generation on actual asphalt roads
"""

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class MilwaukeeRoadNetwork:
    """Manages Milwaukee road network for route generation"""
    
    def __init__(self):
        self.graph = None
        self.graph_projected = None
        self.nodes = None
        self.edges = None
        self._initialized = False
        
    def initialize(self, retries=3):
        """Download and cache Milwaukee road network"""
        if self._initialized:
            return True
        
        for attempt in range(retries):
            try:
                logger.info(f"🌍 Loading Milwaukee road network (attempt {attempt + 1}/{retries})...")
                
                # Download graph (cached automatically by OSMnx in /root/.osmnx)
                self.graph = ox.graph_from_place(
                    "Milwaukee, Wisconsin, USA", 
                    network_type="drive"
                )
                
                # Project to meters for accurate distance calculations
                self.graph_projected = ox.project_graph(self.graph)
                
                # Get nodes and edges as GeoDataFrames
                self.nodes, self.edges = ox.graph_to_gdfs(
                    self.graph, 
                    nodes=True, 
                    edges=True
                )
                
                self._initialized = True
                logger.info(f"✅ Loaded {len(self.edges)} road segments, {len(self.nodes)} nodes")
                return True
                
            except Exception as e:
                logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    import time
                    time.sleep(5)
                else:
                    logger.error("❌ Failed to load road network after all retries")
                    return False
        
        return False
    
    def find_nearest_node(self, lat, lon):
        """Find nearest road node to given coordinates"""
        try:
            node_id = ox.distance.nearest_nodes(
                self.graph, 
                X=lon, 
                Y=lat
            )
            return node_id
        except Exception as e:
            logger.error(f"Error finding nearest node for ({lat}, {lon}): {e}")
            return None
    
    @lru_cache(maxsize=1000)
    def get_route_between_sensors_cached(self, lat1_int, lon1_int, lat2_int, lon2_int):
        """Cached version of route calculation (uses integer coords for hashability)"""
        lat1 = lat1_int / 1000000.0
        lon1 = lon1_int / 1000000.0
        lat2 = lat2_int / 1000000.0
        lon2 = lon2_int / 1000000.0
        
        return self._get_route_between_sensors_impl(lat1, lon1, lat2, lon2)
    
    def get_route_between_sensors(self, lat1, lon1, lat2, lon2):
        """
        Get actual road route between two sensor coordinates.
        Returns list of (lat, lon) coordinates following roads.
        """
        # Convert to integers for caching (round to 6 decimals = ~0.1m precision)
        lat1_int = int(round(lat1 * 1000000))
        lon1_int = int(round(lon1 * 1000000))
        lat2_int = int(round(lat2 * 1000000))
        lon2_int = int(round(lon2 * 1000000))
        
        return self.get_route_between_sensors_cached(lat1_int, lon1_int, lat2_int, lon2_int)
    
    def _get_route_between_sensors_impl(self, lat1, lon1, lat2, lon2):
        """Internal implementation of route calculation"""
        try:
            # Find nearest road nodes
            origin_node = self.find_nearest_node(lat1, lon1)
            dest_node = self.find_nearest_node(lat2, lon2)
            
            if origin_node is None or dest_node is None:
                return None
            
            if origin_node == dest_node:
                # Same node, return single point
                node_data = self.graph.nodes[origin_node]
                return [(node_data['y'], node_data['x'])]
            
            # Calculate shortest path on road network
            try:
                route = ox.shortest_path(
                    self.graph_projected, 
                    origin_node, 
                    dest_node, 
                    weight='length'
                )
            except nx.NetworkXNoPath:
                # No path exists (disconnected roads)
                logger.debug(f"No path between ({lat1},{lon1}) and ({lat2},{lon2})")
                return None
            except Exception as e:
                logger.error(f"Error finding path: {e}")
                return None
            
            if route is None or len(route) < 2:
                return None
            
            # Extract coordinates from route nodes
            route_coords = []
            for node_id in route:
                node_data = self.graph.nodes[node_id]
                route_coords.append((node_data['y'], node_data['x']))  # (lat, lon)
            
            return route_coords
            
        except Exception as e:
            logger.error(f"Error calculating route: {e}")
            return None
    
    def generate_route_segments(self, route_coords):
        """
        Break route into segments for heatmap display.
        
        Args:
            route_coords: List of (lat, lon) tuples along route
            
        Returns:
            List of segment dicts with start/end coordinates
        """
        if not route_coords or len(route_coords) < 2:
            return []
        
        segments = []
        
        for i in range(len(route_coords) - 1):
            start_lat, start_lon = route_coords[i]
            end_lat, end_lon = route_coords[i + 1]
            
            # Calculate center point
            center_lat = (start_lat + end_lat) / 2
            center_lon = (start_lon + end_lon) / 2
            
            # Estimate segment length using Haversine approximation
            # 111,000m per degree latitude, varies for longitude
            lat_diff = abs(end_lat - start_lat)
            lon_diff = abs(end_lon - start_lon)
            
            # Rough distance in meters
            segment_length_m = ((lat_diff * 111000) ** 2 + 
                               (lon_diff * 111000 * abs(lon_diff)) ** 2) ** 0.5
            
            segments.append({
                'segment_start_lat': start_lat,
                'segment_start_lon': start_lon,
                'segment_end_lat': end_lat,
                'segment_end_lon': end_lon,
                'center_lat': center_lat,
                'center_lon': center_lon,
                'segment_length_meters': round(segment_length_m, 2)
            })
        
        return segments
    
    def get_street_name_for_segment(self, lat, lon):
        """Get street name for given coordinates"""
        try:
            nearest_edge = ox.distance.nearest_edges(
                self.graph, 
                X=lon, 
                Y=lat
            )
            
            edge_data = self.edges.loc[nearest_edge]
            street_name = edge_data.get('name', 'Unnamed Road')
            
            # Handle list of names
            if isinstance(street_name, list):
                street_name = street_name[0] if street_name else 'Unnamed Road'
            
            return street_name
            
        except Exception as e:
            logger.debug(f"Could not get street name: {e}")
            return 'Unnamed Road'


# Global singleton instance
road_network = MilwaukeeRoadNetwork()