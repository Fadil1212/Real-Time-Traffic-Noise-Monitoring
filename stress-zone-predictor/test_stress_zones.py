#!/usr/bin/env python3
"""
Test Stress Zone Predictor
Verifies that stress zone predictions are working correctly
"""

import psycopg2
import sys
from datetime import datetime, timedelta, timezone
from tabulate import tabulate

def test_stress_zone_predictor():
    """Test stress zone predictor output"""
    
    print("=" * 80)
    print("🧪 TESTING STRESS ZONE PREDICTOR")
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
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Test 1: Check if table exists
    print("\n📋 Test 1: Check predicted_stress_zones table exists...")
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'predicted_stress_zones'
        );
    """)
    exists = cursor.fetchone()[0]
    
    if exists:
        print("   ✅ Table 'predicted_stress_zones' exists")
    else:
        print("   ❌ Table 'predicted_stress_zones' does not exist")
        conn.close()
        sys.exit(1)
    
    # Test 2: Check if we have predictions
    print("\n📊 Test 2: Check for stress zone predictions...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM predicted_stress_zones;
    """)
    count = cursor.fetchone()[0]
    
    print(f"   Total stress zone records: {count}")
    
    if count == 0:
        print("   ⚠️  No stress zones predicted yet")
        print("   💡 Wait 15 minutes or check: docker logs stress-zone-predictor")
        conn.close()
        sys.exit(0)
    else:
        print(f"   ✅ Found {count} stress zone predictions")
    
    # Test 3: Check recent predictions
    print("\n📈 Test 3: Check recent stress predictions (last hour)...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM predicted_stress_zones
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour';
    """)
    recent_count = cursor.fetchone()[0]
    
    if recent_count > 0:
        print(f"   ✅ Found {recent_count} predictions from last hour")
    else:
        print("   ⚠️  No recent predictions (waiting for next run)")
    
    # Test 4: Display predictions by alert level
    print("\n⚠️  Test 4: Predictions by alert level...")
    cursor.execute("""
        SELECT 
            alert_level,
            COUNT(*) as count,
            ROUND(AVG(predicted_stress_level)::numeric, 1) as avg_stress,
            ROUND(AVG(predicted_noise_contribution)::numeric, 1) as avg_noise
        FROM predicted_stress_zones
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
        GROUP BY alert_level
        ORDER BY 
            CASE alert_level
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'moderate' THEN 3
                ELSE 4
            END;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Alert Level', 'Count', 'Avg Stress', 'Avg Noise (dB)']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    
    # Test 5: Show top stress zones
    print("\n🔥 Test 5: Top predicted stress zones...")
    cursor.execute("""
        SELECT 
            zone_name,
            alert_level,
            ROUND(predicted_stress_level::numeric, 1) as stress,
            ROUND(predicted_noise_contribution::numeric, 1) as noise,
            ROUND(predicted_sentiment_contribution::numeric, 2) as sentiment,
            forecast_timestamp::timestamp(0) as forecast_for
        FROM predicted_stress_zones
        WHERE prediction_timestamp = (
            SELECT MAX(prediction_timestamp) 
            FROM predicted_stress_zones
        )
        AND forecast_timestamp > NOW()
        ORDER BY predicted_stress_level DESC
        LIMIT 10;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Zone', 'Alert', 'Stress', 'Noise (dB)', 'Sentiment', 'Forecast For']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    
    # Test 6: Validate stress scores
    print("\n✅ Test 6: Validate stress scores...")
    cursor.execute("""
        SELECT 
            MIN(predicted_stress_level) as min_stress,
            MAX(predicted_stress_level) as max_stress,
            AVG(predicted_stress_level) as avg_stress
        FROM predicted_stress_zones
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour';
    """)
    
    min_stress, max_stress, avg_stress = cursor.fetchone()
    
    if min_stress and max_stress:
        print(f"   Stress range: {min_stress:.1f} to {max_stress:.1f}")
        print(f"   Average stress: {avg_stress:.1f}")
        
        if 0 <= min_stress <= 100 and 0 <= max_stress <= 100:
            print("   ✅ Stress scores are in valid range (0-100)")
        else:
            print("   ❌ Stress scores are out of valid range!")
    
    # Test 7: Check recommendations
    print("\n💡 Test 7: Sample recommendations...")
    cursor.execute("""
        SELECT 
            zone_name,
            alert_level,
            recommended_action
        FROM predicted_stress_zones
        WHERE prediction_timestamp = (
            SELECT MAX(prediction_timestamp) 
            FROM predicted_stress_zones
        )
        AND alert_level IN ('critical', 'high')
        ORDER BY predicted_stress_level DESC
        LIMIT 3;
    """)
    
    results = cursor.fetchall()
    
    if results:
        print()
        for zone, level, action in results:
            print(f"   📍 {zone} ({level.upper()})")
            print(f"      {action}")
            print()
    else:
        print("   ℹ️  No high-risk zones currently predicted")
    
    # Test 8: Check time distribution
    print("\n⏰ Test 8: Predictions by forecast horizon...")
    cursor.execute("""
        SELECT 
            CASE 
                WHEN forecast_timestamp <= NOW() + INTERVAL '3 hours' THEN '0-3 hours'
                WHEN forecast_timestamp <= NOW() + INTERVAL '6 hours' THEN '3-6 hours'
                WHEN forecast_timestamp <= NOW() + INTERVAL '12 hours' THEN '6-12 hours'
                ELSE '12-24 hours'
            END as time_range,
            COUNT(*) as count,
            COUNT(*) FILTER (WHERE alert_level IN ('critical', 'high')) as high_risk_count
        FROM predicted_stress_zones
        WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
        AND forecast_timestamp > NOW()
        GROUP BY time_range
        ORDER BY 
            CASE time_range
                WHEN '0-3 hours' THEN 1
                WHEN '3-6 hours' THEN 2
                WHEN '6-12 hours' THEN 3
                ELSE 4
            END;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Time Range', 'Total Predictions', 'High Risk Zones']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    
    # Test 9: Check predictor service
    print("\n🐳 Test 9: Check stress zone predictor service...")
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=stress-zone-predictor', '--format', '{{.Status}}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(f"   ✅ Stress zone predictor container is running")
            print(f"      Status: {result.stdout.strip()}")
        else:
            print("   ⚠️  Stress zone predictor container not found")
            print("   💡 Start it with: docker compose up -d stress-zone-predictor")
    except Exception as e:
        print(f"   ⚠️  Could not check Docker status: {e}")
    
    # Test 10: Verify dependency on noise predictor
    print("\n🔗 Test 10: Check dependency on noise predictions...")
    cursor.execute("""
        WITH latest_noise_pred AS (
            SELECT MAX(prediction_timestamp) as ts FROM noise_predictions
        ),
        latest_stress_pred AS (
            SELECT MAX(prediction_timestamp) as ts FROM predicted_stress_zones
        )
        SELECT 
            (SELECT ts FROM latest_noise_pred) as last_noise_prediction,
            (SELECT ts FROM latest_stress_pred) as last_stress_prediction,
            EXTRACT(EPOCH FROM (SELECT ts FROM latest_stress_pred) - (SELECT ts FROM latest_noise_pred)) / 60 as delay_minutes;
    """)
    
    result = cursor.fetchone()
    
    if result and result[0] and result[1]:
        delay = result[2]
        print(f"   Last noise prediction: {result[0]}")
        print(f"   Last stress prediction: {result[1]}")
        print(f"   Delay: {delay:.1f} minutes")
        
        if 0 <= delay <= 5:
            print("   ✅ Stress predictor running shortly after noise predictor")
        elif delay < 0:
            print("   ⚠️  Stress predictor running before noise predictor (unexpected)")
        else:
            print("   ⚠️  Long delay between predictions")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if count > 0:
        print("\n✅ STRESS ZONE PREDICTOR IS WORKING!")
        print(f"\n   Total Predictions: {count}")
        print(f"   Recent Predictions: {recent_count}")
        if avg_stress:
            print(f"   Average Stress Level: {avg_stress:.1f}")
        
        # Alert level summary
        cursor.execute("""
            SELECT alert_level, COUNT(*) 
            FROM predicted_stress_zones
            WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY alert_level;
        """)
        alert_summary = dict(cursor.fetchall())
        
        if alert_summary:
            print(f"\n   Alert Breakdown:")
            print(f"      Critical: {alert_summary.get('critical', 0)}")
            print(f"      High: {alert_summary.get('high', 0)}")
            print(f"      Moderate: {alert_summary.get('moderate', 0)}")
            print(f"      Low: {alert_summary.get('low', 0)}")
    else:
        print("\n⏳ NO STRESS ZONE PREDICTIONS YET")
        print("   1. Make sure noise-predictor ran first")
        print("   2. Wait 15 minutes for first prediction")
        print("   3. Check logs: docker logs stress-zone-predictor")
    
    print("\n" + "=" * 80)
    
    # Close connection
    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        test_stress_zone_predictor()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)