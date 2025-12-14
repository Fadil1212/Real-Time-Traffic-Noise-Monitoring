#!/usr/bin/env python3
"""
Test Correlation Analyzer
Verifies that correlation analysis is working correctly
"""

import psycopg2
import sys
from datetime import datetime, timedelta
from tabulate import tabulate

def test_correlation_analyzer():
    """Test correlation analyzer output"""
    
    print("=" * 80)
    print("🧪 TESTING CORRELATION ANALYZER")
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
        print("\n💡 If running from host, make sure PostgreSQL port is exposed")
        sys.exit(1)
    
    cursor = conn.cursor()
    
    # Test 1: Check if correlation table exists
    print("\n📋 Test 1: Check correlation table exists...")
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'noise_sentiment_correlation'
        );
    """)
    exists = cursor.fetchone()[0]
    
    if exists:
        print("   ✅ Table 'noise_sentiment_correlation' exists")
    else:
        print("   ❌ Table 'noise_sentiment_correlation' does not exist")
        conn.close()
        sys.exit(1)
    
    # Test 2: Check if we have correlation data
    print("\n📊 Test 2: Check for correlation data...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM noise_sentiment_correlation;
    """)
    count = cursor.fetchone()[0]
    
    print(f"   Total correlation records: {count}")
    
    if count == 0:
        print("   ⚠️  No correlation data found yet")
        print("   💡 Correlation analyzer may not have run yet")
        print("   💡 Wait 5 minutes or check: docker logs correlation-analyzer")
        conn.close()
        sys.exit(0)
    else:
        print(f"   ✅ Found {count} correlation records")
    
    # Test 3: Check recent correlations
    print("\n📈 Test 3: Check recent correlations (last hour)...")
    cursor.execute("""
        SELECT COUNT(*) 
        FROM noise_sentiment_correlation
        WHERE analysis_timestamp > NOW() - INTERVAL '1 hour';
    """)
    recent_count = cursor.fetchone()[0]
    
    if recent_count > 0:
        print(f"   ✅ Found {recent_count} correlations from last hour")
    else:
        print("   ⚠️  No recent correlations (may be waiting for next run)")
    
    # Test 4: Display latest correlations
    print("\n📊 Test 4: Display latest correlations...")
    cursor.execute("""
        SELECT 
            location_name,
            ROUND(correlation_coefficient::numeric, 3) as correlation,
            ROUND(avg_noise_level::numeric, 1) as avg_noise,
            ROUND(avg_sentiment_score::numeric, 2) as avg_sentiment,
            sample_size,
            time_window,
            analysis_timestamp::timestamp(0) as analyzed_at
        FROM noise_sentiment_correlation
        ORDER BY analysis_timestamp DESC
        LIMIT 10;
    """)
    
    results = cursor.fetchall()
    
    if results:
        headers = ['Location', 'Correlation (r)', 'Avg Noise (dB)', 'Avg Sentiment', 
                   'Sample Size', 'Time Window', 'Analyzed At']
        print()
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()
    else:
        print("   ⚠️  No correlation data to display")
    
    # Test 5: Validate correlation values
    print("\n✅ Test 5: Validate correlation values...")
    cursor.execute("""
        SELECT 
            MIN(correlation_coefficient) as min_corr,
            MAX(correlation_coefficient) as max_corr,
            AVG(correlation_coefficient) as avg_corr
        FROM noise_sentiment_correlation;
    """)
    
    min_corr, max_corr, avg_corr = cursor.fetchone()
    
    if min_corr and max_corr:
        print(f"   Correlation range: {min_corr:.3f} to {max_corr:.3f}")
        print(f"   Average correlation: {avg_corr:.3f}")
        
        # Check if correlations are in valid range
        if -1.0 <= min_corr <= 1.0 and -1.0 <= max_corr <= 1.0:
            print("   ✅ Correlations are in valid range (-1 to +1)")
        else:
            print("   ❌ Correlations are out of valid range!")
        
        # Check if we have expected negative correlation
        if avg_corr < -0.5:
            print("   ✅ Strong negative correlation detected (expected!)")
            print("      Higher noise → More negative sentiment ✅")
        elif avg_corr < 0:
            print("   ✅ Negative correlation detected (expected!)")
        else:
            print("   ⚠️  Positive or no correlation (unexpected)")
    
    # Test 6: Check by location types
    print("\n📍 Test 6: Correlation by location type...")
    cursor.execute("""
        SELECT 
            location_name,
            ROUND(AVG(correlation_coefficient)::numeric, 3) as avg_correlation,
            COUNT(*) as count
        FROM noise_sentiment_correlation
        GROUP BY location_name
        ORDER BY avg_correlation;
    """)
    
    location_results = cursor.fetchall()
    
    if location_results:
        headers = ['Location', 'Avg Correlation', 'Count']
        print()
        print(tabulate(location_results, headers=headers, tablefmt='grid'))
        print()
    
    # Test 7: Check statistical significance
    print("\n📊 Test 7: Check statistical significance...")
    cursor.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE statistical_significance < 0.001) as very_significant,
            COUNT(*) FILTER (WHERE statistical_significance < 0.01) as significant,
            COUNT(*) FILTER (WHERE statistical_significance < 0.05) as moderately_significant,
            COUNT(*) FILTER (WHERE statistical_significance >= 0.05) as not_significant
        FROM noise_sentiment_correlation;
    """)
    
    very_sig, sig, mod_sig, not_sig = cursor.fetchone()
    
    print(f"   Very Significant (p < 0.001):      {very_sig}")
    print(f"   Significant (p < 0.01):            {sig}")
    print(f"   Moderately Significant (p < 0.05): {mod_sig}")
    print(f"   Not Significant (p >= 0.05):       {not_sig}")
    
    if very_sig + sig > 0:
        print("   ✅ Found statistically significant correlations!")
    
    # Test 8: Check correlation analyzer service
    print("\n🐳 Test 8: Check correlation analyzer service...")
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=correlation-analyzer', '--format', '{{.Status}}'],
            capture_output=True,
            text=True
        )
        
        if result.stdout.strip():
            print(f"   ✅ Correlation analyzer container is running")
            print(f"      Status: {result.stdout.strip()}")
        else:
            print("   ⚠️  Correlation analyzer container not found")
            print("   💡 Start it with: ./start.sh → 1")
    except Exception as e:
        print(f"   ⚠️  Could not check Docker status: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if count > 0 and avg_corr and avg_corr < 0:
        print("\n✅ CORRELATION ANALYZER IS WORKING!")
        print(f"\n   Total Records: {count}")
        print(f"   Average Correlation: {avg_corr:.3f}")
        print(f"   Recent Records (1h): {recent_count}")
        print(f"   Significant Results: {very_sig + sig}")
        print("\n   Interpretation:")
        if avg_corr < -0.7:
            print("   🎯 Strong negative correlation")
            print("      High noise strongly correlates with negative sentiment")
        elif avg_corr < -0.5:
            print("   🎯 Moderate negative correlation")
            print("      High noise moderately correlates with negative sentiment")
        else:
            print("   🎯 Weak negative correlation")
            print("      Some relationship between noise and sentiment")
    elif count > 0:
        print("\n⚠️  CORRELATION DATA EXISTS BUT MAY BE UNUSUAL")
        print(f"   Average correlation: {avg_corr:.3f}")
        print("   Expected: Strong negative correlation (< -0.5)")
    else:
        print("\n⏳ NO CORRELATION DATA YET")
        print("   Wait 5 minutes for first analysis")
        print("   Or check logs: docker logs correlation-analyzer")
    
    print("\n" + "=" * 80)
    
    # Close connection
    cursor.close()
    conn.close()


if __name__ == "__main__":
    try:
        test_correlation_analyzer()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)