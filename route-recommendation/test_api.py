#!/usr/bin/env python3
"""
Test Alternative Routes API
Tests the route recommendation endpoint
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("=" * 70)
    print("🏥 Testing Health Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print("✅ Health check passed!\n")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}\n")
        return False

def test_get_locations():
    """Test get locations endpoint"""
    print("=" * 70)
    print("📍 Testing Get Locations Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/api/locations", timeout=10)
        print(f"Status Code: {response.status_code}")
        
        data = response.json()
        print(f"Found {data.get('count', 0)} locations")
        
        if data.get('locations'):
            print("\nSample locations:")
            for loc in data['locations'][:5]:
                print(f"  - {loc['street']}, {loc['neighborhood']}")
        
        print("✅ Get locations passed!\n")
        return data.get('locations', [])
    except Exception as e:
        print(f"❌ Get locations failed: {e}\n")
        return []

def test_route_recommendations(origin, destination):
    """Test route recommendations endpoint"""
    print("=" * 70)
    print("🗺️  Testing Route Recommendations")
    print("=" * 70)
    print(f"Origin: {origin}")
    print(f"Destination: {destination}\n")
    
    try:
        payload = {
            "origin": origin,
            "destination": destination
        }
        
        response = requests.post(
            f"{API_URL}/api/route-recommendations",
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        data = response.json()
        
        if data.get('success'):
            print(f"\n✅ Found {len(data.get('routes', []))} alternative routes:\n")
            
            for route in data.get('routes', []):
                print(f"{'='*70}")
                print(f"Route {route['rank']}: {route['type'].upper()}")
                print(f"{'='*70}")
                print(f"Path: {' → '.join(route['path'][:5])}...")
                print(f"Distance: {route['distance_meters']:.0f} meters")
                print(f"Time: ~{route['estimated_time_minutes']:.1f} minutes")
                print(f"Noise Level: {route['avg_noise_level']:.1f} dB ({route['noise_category']})")
                print(f"Recommendation: {route['recommendation']}")
                print()
            
            print("✅ Route recommendations test passed!\n")
            return True
        else:
            print(f"❌ API returned error: {data.get('error')}\n")
            return False
            
    except Exception as e:
        print(f"❌ Route recommendations test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_invalid_request():
    """Test with invalid request"""
    print("=" * 70)
    print("🚫 Testing Invalid Request Handling")
    print("=" * 70)
    
    try:
        payload = {}  # Missing origin and destination
        
        response = requests.post(
            f"{API_URL}/api/route-recommendations",
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            print("✅ Correctly rejected invalid request!\n")
            return True
        else:
            print("❌ Should have returned 400 Bad Request\n")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("🧪 ALTERNATIVE ROUTES API - TEST SUITE")
    print("=" * 70 + "\n")
    
    # Test 1: Health check
    if not test_health():
        print("❌ API is not running. Please start it first.")
        print("   docker compose up -d alternative-routes-api")
        return
    
    # Test 2: Get locations
    locations = test_get_locations()
    
    if not locations:
        print("⚠️  No locations available. Make sure route segments exist in the database.")
        return
    
    # Test 3: Route recommendations with real data
    if len(locations) >= 2:
        origin = locations[0]['street']
        destination = locations[-1]['street']
        test_route_recommendations(origin, destination)
    else:
        # Fallback test with generic names
        test_route_recommendations("Mewlid Road", "Railway Station Road")
    
    # Test 4: Invalid request
    test_invalid_request()
    
    print("=" * 70)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()