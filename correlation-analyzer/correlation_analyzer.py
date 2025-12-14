#!/usr/bin/env python3
"""
Noise-Sentiment Correlation Analysis
Runs every 5 minutes to analyze correlation between noise levels and sentiment scores

Analyzes:
- Pearson correlation coefficient
- Patterns by location, time of day, day of week
- Statistical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
import psycopg2
from datetime import datetime, timedelta
import logging
import time
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """Analyzes correlation between noise levels and sentiment scores"""
    
    def __init__(self, db_config):
        """
        Initialize analyzer
        
        Args:
            db_config: Database connection parameters
        """
        self.db_config = db_config
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False
    
    def load_recent_data(self, hours=1):
        """
        Load recent noise and sentiment data
        
        Args:
            hours: Number of hours of data to analyze
            
        Returns:
            DataFrame with noise and sentiment data
        """
        logger.info(f"📖 Loading data from last {hours} hour(s)...")
        
        query = f"""
        SELECT 
            nr.timestamp,
            nr.sensor_id,
            nr.street_name,
            nr.neighborhood,
            nr.latitude,
            nr.longitude,
            nr.noise_level,
            nr.sentiment_text,
            EXTRACT(HOUR FROM nr.timestamp) as hour_of_day,
            EXTRACT(DOW FROM nr.timestamp) as day_of_week,
            CASE 
                WHEN EXTRACT(DOW FROM nr.timestamp) IN (0, 6) THEN 'Weekend'
                ELSE 'Weekday'
            END as day_type,
            CASE
                WHEN EXTRACT(HOUR FROM nr.timestamp) BETWEEN 6 AND 9 THEN 'Morning Rush'
                WHEN EXTRACT(HOUR FROM nr.timestamp) BETWEEN 12 AND 14 THEN 'Lunch'
                WHEN EXTRACT(HOUR FROM nr.timestamp) BETWEEN 17 AND 19 THEN 'Evening Rush'
                WHEN EXTRACT(HOUR FROM nr.timestamp) BETWEEN 22 AND 5 THEN 'Night'
                ELSE 'Normal'
            END as time_period
        FROM noise_readings nr
        WHERE nr.timestamp > NOW() - INTERVAL '{hours} hours'
        AND nr.sentiment_text IS NOT NULL
        ORDER BY nr.timestamp DESC
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} records")
        
        return df
    
    def analyze_sentiment(self, df):
        """
        Analyze sentiment from text using DistilBERT model
        
        Args:
            df: DataFrame with sentiment_text column
            
        Returns:
            DataFrame with sentiment_score added
        """
        logger.info("🔍 Analyzing sentiment with DistilBERT model...")
        
        # Import sentiment analyzer
        try:
            from utils.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer(model_path="models/sentiment")
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            logger.error("   Make sure model is downloaded: ./start.sh → 18")
            raise
        
        # Analyze in batches for efficiency
        texts = df['sentiment_text'].fillna("").tolist()
        
        # Batch analyze (much faster than one-by-one)
        try:
            results = analyzer.batch_analyze(texts)
            df['sentiment_score'] = [r['sentiment_score'] for r in results]
            df['sentiment_label'] = [r['label'] for r in results]
            df['sentiment_confidence'] = [r['confidence'] for r in results]
        except Exception as e:
            logger.error(f"❌ Batch analysis failed: {e}")
            # Fallback to individual analysis
            logger.info("   Falling back to individual analysis...")
            scores = []
            for text in texts:
                if text:
                    result = analyzer.analyze(text)
                    scores.append(result['sentiment_score'])
                else:
                    scores.append(0.0)
            df['sentiment_score'] = scores
        
        logger.info(f"✅ Sentiment analysis complete")
        logger.info(f"   Negative: {(df['sentiment_score'] < -0.3).sum()}")
        logger.info(f"   Neutral: {(df['sentiment_score'].abs() <= 0.3).sum()}")
        logger.info(f"   Positive: {(df['sentiment_score'] > 0.3).sum()}")
        logger.info(f"   Avg Sentiment: {df['sentiment_score'].mean():.3f}")
        
        return df
    
    def calculate_correlation(self, df, group_by=None):
        """
        Calculate Pearson correlation between noise and sentiment
        
        Args:
            df: DataFrame with noise_level and sentiment_score
            group_by: Column(s) to group by (e.g., 'neighborhood', 'time_period')
            
        Returns:
            DataFrame with correlation results
        """
        results = []
        
        if group_by:
            groups = df.groupby(group_by)
        else:
            groups = [('Overall', df)]
        
        for group_name, group_df in groups:
            if len(group_df) < 10:
                logger.warning(f"⚠️  Skipping {group_name}: not enough data ({len(group_df)} samples)")
                continue
            
            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(
                group_df['noise_level'],
                group_df['sentiment_score']
            )
            
            # Calculate additional statistics
            avg_noise = group_df['noise_level'].mean()
            avg_sentiment = group_df['sentiment_score'].mean()
            sample_size = len(group_df)
            
            # Determine significance
            if p_value < 0.001:
                significance = 'Very Significant'
            elif p_value < 0.01:
                significance = 'Significant'
            elif p_value < 0.05:
                significance = 'Moderately Significant'
            else:
                significance = 'Not Significant'
            
            result = {
                'group': str(group_name),
                'correlation': round(correlation, 4),
                'p_value': round(p_value, 4),
                'significance': significance,
                'avg_noise': round(avg_noise, 2),
                'avg_sentiment': round(avg_sentiment, 2),
                'sample_size': sample_size
            }
            
            results.append(result)
            
            logger.info(f"📊 {group_name}:")
            logger.info(f"   Correlation: {correlation:.4f} (p={p_value:.4f})")
            logger.info(f"   Avg Noise: {avg_noise:.1f} dB")
            logger.info(f"   Avg Sentiment: {avg_sentiment:.2f}")
            logger.info(f"   Sample Size: {sample_size}")
        
        return pd.DataFrame(results)
    
    def store_correlations(self, correlations_df, time_window):
        """
        Store correlation results in database
        
        Args:
            correlations_df: DataFrame with correlation results
            time_window: Time window analyzed (e.g., '1 hour', '24 hours')
        """
        logger.info("💾 Storing correlation results...")
        
        cursor = self.conn.cursor()
        
        for _, row in correlations_df.iterrows():
            # Parse group information
            group_str = row['group']
            
            # Determine location details
            if ' - ' in group_str:
                parts = group_str.split(' - ')
                location_name = parts[0]
            else:
                location_name = group_str
            
            # Insert into database
            query = """
            INSERT INTO noise_sentiment_correlation (
                analysis_timestamp,
                time_window,
                location_name,
                avg_noise_level,
                avg_sentiment_score,
                correlation_coefficient,
                sample_size,
                statistical_significance
            ) VALUES (
                NOW(),
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """
            
            cursor.execute(query, (
                time_window,
                location_name,
                row['avg_noise'],
                row['avg_sentiment'],
                row['correlation'],
                row['sample_size'],
                row['p_value']
            ))
        
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Stored {len(correlations_df)} correlation results")
    
    def run_analysis(self, hours=1):
        """
        Run complete correlation analysis
        
        Args:
            hours: Hours of data to analyze
        """
        logger.info("=" * 80)
        logger.info("🔬 CORRELATION ANALYSIS")
        logger.info("=" * 80)
        
        try:
            # Load data
            df = self.load_recent_data(hours)
            
            if len(df) < 10:
                logger.warning("⚠️  Not enough data for analysis (need 10+ samples)")
                return
            
            # Analyze sentiment
            df = self.analyze_sentiment(df)
            
            # Overall correlation
            logger.info("\n📊 Overall Correlation:")
            overall = self.calculate_correlation(df)
            
            # By neighborhood
            logger.info("\n📍 By Neighborhood:")
            by_neighborhood = self.calculate_correlation(df, group_by='neighborhood')
            
            # By time period
            logger.info("\n⏰ By Time Period:")
            by_time = self.calculate_correlation(df, group_by='time_period')
            
            # By day type
            logger.info("\n📅 By Day Type:")
            by_day_type = self.calculate_correlation(df, group_by='day_type')
            
            # Combine all results
            all_correlations = pd.concat([
                overall,
                by_neighborhood,
                by_time,
                by_day_type
            ], ignore_index=True)
            
            # Store in database
            time_window = f"{hours} hour{'s' if hours > 1 else ''}"
            self.store_correlations(all_correlations, time_window)
            
            logger.info("=" * 80)
            logger.info("✅ CORRELATION ANALYSIS COMPLETE!")
            logger.info("=" * 80)
            
            # Print summary
            logger.info("\n📈 Summary:")
            logger.info(f"   Total Samples: {len(df)}")
            logger.info(f"   Avg Noise: {df['noise_level'].mean():.1f} dB")
            logger.info(f"   Avg Sentiment: {df['sentiment_score'].mean():.2f}")
            logger.info(f"   Overall Correlation: {overall['correlation'].values[0]:.4f}")
            logger.info(f"   Significant Groups: {(all_correlations['p_value'] < 0.05).sum()}")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("🔌 Database connection closed")


def main():
    """Main function - runs every 5 minutes"""
    
    # Database configuration
    DB_CONFIG = {
        'host': 'localhost',  # Use 'postgres' if running in Docker
        'port': 5432,
        'database': 'traffic_noise_db',
        'user': 'traffic_user',
        'password': 'traffic_pass'
    }
    
    # Check if running in Docker
    import os
    if os.path.exists('/.dockerenv'):
        DB_CONFIG['host'] = 'postgres'
    
    # Configuration
    RUN_INTERVAL = 300  # 5 minutes in seconds
    ANALYSIS_WINDOW = 1  # Analyze last 1 hour
    
    logger.info("=" * 80)
    logger.info("Meliwakie Traffic Noise - Correlation Analysis Service")
    logger.info("=" * 80)
    logger.info(f"⏱️  Run Interval: Every {RUN_INTERVAL/60} minutes")
    logger.info(f"📊 Analysis Window: Last {ANALYSIS_WINDOW} hour(s)")
    logger.info(f"🗄️  Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info("=" * 80)
    
    # Create analyzer
    analyzer = CorrelationAnalyzer(DB_CONFIG)
    
    if not analyzer.connect():
        logger.error("❌ Failed to connect to database. Exiting.")
        sys.exit(1)
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run once and exit
            logger.info("🎯 Running single analysis...")
            analyzer.run_analysis(ANALYSIS_WINDOW)
        else:
            # Run continuously
            logger.info("🔄 Starting continuous analysis...")
            logger.info("   Press Ctrl+C to stop")
            
            while True:
                analyzer.run_analysis(ANALYSIS_WINDOW)
                
                logger.info(f"\n⏸️  Waiting {RUN_INTERVAL/60} minutes until next analysis...")
                time.sleep(RUN_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping correlation analysis (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        analyzer.close()
        logger.info("✅ Correlation analysis service stopped")


if __name__ == "__main__":
    main()