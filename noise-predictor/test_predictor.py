#!/usr/bin/env python3
"""
Test Noise Predictor
Verifies that noise predictions are working correctly
"""

import psycopg2
import sys
from datetime import datetime, timedelta
from tabulate import tabulate

def test_noise_predictor():
    """Test noise predictor output"""
    
    print("=" * 80)
    print("🧪 TESTING NOISE PREDICTOR")
    print("=" * 80)
    print()
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="traffic_noise_db",
            user="traffic_user",
            password="traffic_pass"
        )
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        print("\n💡 Make sure PostgreSQL port is exposed and accessible")
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Test 1: Check if predictions table exists
    print("\n📋 Test 1: Check predictions table exists...")
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'noise_predictions'
        );
    """)
    exists = cursor.fetchone()[0]
    
    if exists:
        print("   ✅ Table 'noise_predictions' exists")
    else:
        print("   ❌ Table 'noise_predictions' does not exist")
        conn.close()
        sys.exit(1)
    
    # Test 2: Check if we have predictions
    print("\n📊 Test 2: Check for prediction data...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM noise_predictions;
    """)
    count = cursor.fetchone()[0]
    
    print(f"   Total prediction records: {count}")
    
    if count == 0:
        print("   ⚠️  No predictions found yet")
        print("   💡 Predictor may not have run yet")
        print("   💡 Wait 15 minutes or check: docker logs noise-predictor")
        conn.close()
        sys.exit(0)
    else:
        print(f"   ✅ Found {count} prediction records")
    
    # Test 3: Check recent predictions
    print("\n📈 Test 3: Check recent predictions (last hour)...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM noise_predictions
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour';
    """)
    recent_count = cursor.fetchone()[0]
    
    if recent_count > 0:
        print(f"   ✅ Found {recent_count} predictions from last hour")
    else:
        print("   ⚠️  No recent predictions (waiting for next run)")
    
    # Test 4: Display latest predictions by forecast horizon
    print("\n🔮 Test 4: Display predictions by forecast horizon...")
    cursor.execute("""
        SELECT 
            forecast_horizon,
            COUNT(*) as prediction_count,
            ROUND(AVG(predicted_noise_level)::numeric, 1) as avg_predicted_noise,
            ROUND(MIN(predicted_noise_level)::numeric, 1) as min_noise,
            ROUND(MAX(predicted_noise_level)::numeric, 1) as max_noise
        FROM noise_predictions
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY forecast_horizon
        ORDER BY forecast_horizon;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Forecast (hours)', 'Count', 'Avg Noise (dB)', 'Min (dB)', 'Max (dB)']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    else:
        print("   ⚠️  No predictions to display")
    
    # Test 5: Show sample predictions for each location
    print("\n📍 Test 5: Sample predictions by location...")
    cursor.execute("""
        SELECT 
            street_name,
            neighborhood,
            forecast_horizon,
            ROUND(predicted_noise_level::numeric, 1) as predicted_noise,
            forecast_timestamp::timestamp(0) as forecast_for
        FROM noise_predictions
        WHERE prediction_timestamp = (
            SELECT MAX(prediction_timestamp) 
            FROM noise_predictions
        )
        ORDER BY street_name, forecast_horizon
        LIMIT 15;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Street', 'Neighborhood', 'Hours Ahead', 'Predicted (dB)', 'Forecast For']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    
    # Test 6: Validate prediction values
    print("\n✅ Test 6: Validate prediction values...")
    cursor.execute("""
        SELECT 
            MIN(predicted_noise_level) as min_pred,
            MAX(predicted_noise_level) as max_pred,
            AVG(predicted_noise_level) as avg_pred
        FROM noise_predictions
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour';
    """)
    
    min_pred, max_pred, avg_pred = cursor.fetchone()
    
    if min_pred and max_pred:
        print(f"   Prediction range: {min_pred:.1f} to {max_pred:.1f} dB")
        print(f"   Average prediction: {avg_pred:.1f} dB")
        
        # Check if predictions are in valid range
        if 40 <= min_pred <= 110 and 40 <= max_pred <= 110:
            print("   ✅ Predictions are in valid range (40-110 dB)")
        else:
            print("   ❌ Predictions are out of valid range!")
    
    # Test 7: Check confidence intervals
    print("\n📊 Test 7: Check confidence intervals...")
    cursor.execute("""
        SELECT 
            AVG(prediction_interval_upper - prediction_interval_lower) as avg_interval_width,
            AVG(confidence_score) as avg_confidence
        FROM noise_predictions
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour';
    """)
    
    avg_width, avg_confidence = cursor.fetchone()
    
    if avg_width:
        print(f"   Average interval width: ±{avg_width/2:.1f} dB")
        print(f"   Average confidence: {avg_confidence:.2%}")
        
        if avg_width <= 15:
            print("   ✅ Confidence intervals are reasonable")
    
    # Test 8: Compare predictions vs actual (if available)
    print("\n🎯 Test 8: Compare predictions vs actual...")
    cursor.execute("""
        WITH recent_predictions AS (
            SELECT 
                street_name,
                neighborhood,
                forecast_timestamp,
                predicted_noise_level
            FROM noise_predictions
            WHERE forecast_timestamp BETWEEN NOW() - INTERVAL '1 hour' AND NOW()
        ),
        actual_readings AS (
            SELECT 
                street_name,
                neighborhood,
                timestamp,
                AVG(noise_level) as actual_noise
            FROM noise_readings
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY street_name, neighborhood, timestamp
        )
        SELECT 
            p.street_name,
            ROUND(AVG(p.predicted_noise_level)::numeric, 1) as avg_predicted,
            ROUND(AVG(a.actual_noise)::numeric, 1) as avg_actual,
            ROUND(AVG(ABS(p.predicted_noise_level - a.actual_noise))::numeric, 1) as mae
        FROM recent_predictions p
        JOIN actual_readings a 
            ON p.street_name = a.street_name 
            AND p.neighborhood = a.neighborhood
            AND p.forecast_timestamp::date = a.timestamp::date
            AND EXTRACT(HOUR FROM p.forecast_timestamp) = EXTRACT(HOUR FROM a.timestamp)
        GROUP BY p.street_name
        LIMIT 5;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Street', 'Predicted (dB)', 'Actual (dB)', 'MAE (dB)']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
        
        # Calculate overall MAE
        mae_values = [r[3] for r in results]
        avg_mae = sum(mae_values) / len(mae_values)
        
        print(f"   Average MAE: {avg_mae:.1f} dB")
        
        if avg_mae < 5:
            print("   ✅ Excellent prediction accuracy!")
        elif avg_mae < 10:
            print("   ✅ Good prediction accuracy")
        else:
            print("   ⚠️  Predictions could be improved")
    else:
        print("   ⏳ Not enough data yet to compare predictions vs actual")
    
    # Test 9: Check predictor service
    print("\n🐳 Test 9: Check predictor service...")
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=noise-predictor', '--format', '{{.Status}}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(f"   ✅ Noise predictor container is running")
            print(f"      Status: {result.stdout.strip()}")
        else:
            print("   ⚠️  Noise predictor container not found")
            print("   💡 Start it with: docker compose up -d noise-predictor")
    except Exception as e:
        print(f"   ⚠️  Could not check Docker status: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if count > 0:
        print("\n✅ NOISE PREDICTOR IS WORKING!")
        print(f"\n   Total Predictions: {count}")
        print(f"   Recent Predictions: {recent_count}")
        if avg_pred:
            print(f"   Average Predicted Noise: {avg_pred:.1f} dB")
        print(f"   Forecast Horizons: 1h, 3h, 6h, 12h, 24h")
        
        if results and avg_mae:
            print(f"\n   Prediction Accuracy (MAE): {avg_mae:.1f} dB")
    else:
        print("\n⏳ NO PREDICTIONS YET")
        print("   Wait 15 minutes for first prediction")
        print("   Or check logs: docker logs noise-predictor")
    
    print("\n" + "=" * 80)
    
    # Close connection
    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        test_noise_predictor()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)