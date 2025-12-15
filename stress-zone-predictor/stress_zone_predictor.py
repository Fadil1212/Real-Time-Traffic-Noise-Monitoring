#!/usr/bin/env python3
"""
Stress Zone Predictor Service
Uses noise predictions + sentiment analysis to identify future stress zones
Runs every 15 minutes (after noise predictor)
"""

import pandas as pd
import numpy as np
import psycopg2
import logging
from datetime import datetime, timedelta, timezone
import time
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StressZonePredictor:
    """Predicts future stress zones based on noise predictions and sentiment"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.sentiment_analyzer = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    def load_sentiment_analyzer(self):
        """Load sentiment analyzer"""
        try:
            # Import from utils
            import sys
            sys.path.append('/app')
            from utils.sentiment_analyzer import SentimentAnalyzer
            
            self.sentiment_analyzer = SentimentAnalyzer(model_path="/app/models/sentiment")
            logger.info("✅ Sentiment analyzer loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment analyzer: {e}")
            logger.error("   Will use fallback sentiment calculation")
            return False
    
    def get_recent_predictions(self):
        """Get recent noise predictions"""
        query = """
        SELECT 
            prediction_timestamp,
            forecast_timestamp,
            street_name,
            neighborhood,
            latitude,
            longitude,
            predicted_noise_level,
            forecast_horizon
        FROM noise_predictions
        WHERE prediction_timestamp = (
            SELECT MAX(prediction_timestamp) 
            FROM noise_predictions
        )
        AND forecast_timestamp > NOW()
        ORDER BY forecast_timestamp
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} predictions")
        
        return df
    
    def generate_sentiment_description(self, noise_level, street_name, neighborhood):
        """Generate sentiment description based on noise level"""
        if noise_level >= 90:
            return f"Heavy traffic on {street_name} in {neighborhood}. Very loud and congested area with significant noise pollution causing major discomfort."
        elif noise_level >= 80:
            return f"Busy route through {street_name}, {neighborhood}. High traffic noise expected. Uncomfortable environment for pedestrians and residents."
        elif noise_level >= 70:
            return f"Moderate traffic conditions on {street_name} in {neighborhood}. Noticeable noise levels but generally acceptable."
        else:
            return f"Quiet route through {street_name}, {neighborhood}. Low traffic and peaceful environment. Pleasant area."
    
    def analyze_sentiment(self, text):
        """Analyze sentiment with fallback"""
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer.analyze(text)
                return result['sentiment_score']
            except Exception as e:
                logger.warning(f"⚠️  Sentiment analysis failed: {e}, using fallback")
        
        # Fallback: estimate from keywords
        text_lower = text.lower()
        
        negative_keywords = ['loud', 'heavy', 'congested', 'uncomfortable', 'major', 'significant']
        positive_keywords = ['quiet', 'peaceful', 'pleasant', 'low', 'acceptable']
        
        neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
        pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
        
        # Scale to -1 to 1
        if neg_count > pos_count:
            return -0.7
        elif pos_count > neg_count:
            return 0.5
        else:
            return 0.0
    
    def calculate_stress_score(self, noise_level, sentiment_score):
        """
        Calculate stress score (0-100)
        Formula: 0.7 * noise_weight + 0.3 * sentiment_weight
        """
        # Normalize noise to 0-100 (40-100 dB range)
        noise_weight = ((noise_level - 40) / 60) * 100
        noise_weight = max(0, min(100, noise_weight))
        
        # Normalize sentiment to 0-100 (inverse: -1 is high stress, +1 is low stress)
        sentiment_weight = ((1 - sentiment_score) / 2) * 100
        sentiment_weight = max(0, min(100, sentiment_weight))
        
        # Combined stress score
        stress_score = (0.7 * noise_weight) + (0.3 * sentiment_weight)
        
        return round(stress_score, 2)
    
    def get_alert_level(self, stress_score):
        """Determine alert level"""
        if stress_score >= 85:
            return 'critical'
        elif stress_score >= 70:
            return 'high'
        elif stress_score >= 50:
            return 'moderate'
        else:
            return 'low'
    
    def get_recommended_action(self, alert_level, street_name, forecast_time):
        """Generate recommended action"""
        time_str = forecast_time.strftime('%I:%M %p')
        
        if alert_level == 'critical':
            return f"⛔ AVOID {street_name} around {time_str}. Seek alternative routes. Expected very high noise and stress levels."
        elif alert_level == 'high':
            return f"⚠️ Consider alternative routes to {street_name} at {time_str}. High stress levels expected."
        elif alert_level == 'moderate':
            return f"ℹ️ Be aware: Moderate noise levels expected on {street_name} at {time_str}."
        else:
            return f"✅ {street_name} is expected to be relatively quiet at {time_str}."
    
    def predict_stress_zones(self, predictions_df):
        """Predict stress zones from noise predictions"""
        stress_zones = []
        
        logger.info(f"🔍 Analyzing {len(predictions_df)} predictions for stress...")
        
        for idx, row in predictions_df.iterrows():
            # Generate sentiment description
            description = self.generate_sentiment_description(
                row['predicted_noise_level'],
                row['street_name'],
                row['neighborhood']
            )
            
            # Analyze sentiment
            sentiment_score = self.analyze_sentiment(description)
            
            # Calculate stress score
            stress_score = self.calculate_stress_score(
                row['predicted_noise_level'],
                sentiment_score
            )
            
            # Determine alert level
            alert_level = self.get_alert_level(stress_score)
            
            # Only store moderate and above
            if stress_score >= 50:
                zone = {
                    'prediction_timestamp': row['prediction_timestamp'],
                    'forecast_timestamp': row['forecast_timestamp'],
                    'zone_name': f"{row['street_name']} - {row['neighborhood']}",
                    'street_name': row['street_name'],
                    'neighborhood': row['neighborhood'],
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'predicted_stress_level': stress_score,
                    'predicted_noise_contribution': round(row['predicted_noise_level'], 2),
                    'predicted_sentiment_contribution': round(sentiment_score, 2),
                    'alert_level': alert_level,
                    'recommended_action': self.get_recommended_action(
                        alert_level,
                        row['street_name'],
                        row['forecast_timestamp']
                    )
                }
                
                stress_zones.append(zone)
        
        # Sort by stress level (highest first)
        stress_zones.sort(key=lambda x: x['predicted_stress_level'], reverse=True)
        
        logger.info(f"✅ Identified {len(stress_zones)} stress zones")
        
        # Log top 3 stress zones
        if stress_zones:
            logger.info("\n📊 Top Predicted Stress Zones:")
            for i, zone in enumerate(stress_zones[:3], 1):
                hours_ahead = (zone['forecast_timestamp'] - datetime.now(timezone.utc)).total_seconds() / 3600
                logger.info(f"   {i}. {zone['zone_name']}")
                logger.info(f"      Stress: {zone['predicted_stress_level']:.1f} ({zone['alert_level'].upper()})")
                logger.info(f"      Noise: {zone['predicted_noise_contribution']:.1f} dB")
                logger.info(f"      Time: {zone['forecast_timestamp'].strftime('%I:%M %p')} (+{hours_ahead:.1f}h)")
        
        return stress_zones
    
    def store_stress_zones(self, stress_zones):
        """Store predicted stress zones in database"""
        if not stress_zones:
            logger.warning("⚠️  No stress zones to store")
            return
        
        logger.info(f"💾 Storing {len(stress_zones)} stress zones...")
        
        cursor = self.conn.cursor()
        
        for zone in stress_zones:
            query = """
            INSERT INTO predicted_stress_zones (
                prediction_timestamp,
                forecast_timestamp,
                zone_name,
                street_name,
                neighborhood,
                latitude,
                longitude,
                predicted_stress_level,
                predicted_noise_contribution,
                predicted_sentiment_contribution,
                alert_level,
                recommended_action
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            cursor.execute(query, (
                zone['prediction_timestamp'],
                zone['forecast_timestamp'],
                zone['zone_name'],
                zone['street_name'],
                zone['neighborhood'],
                zone['latitude'],
                zone['longitude'],
                zone['predicted_stress_level'],
                zone['predicted_noise_contribution'],
                zone['predicted_sentiment_contribution'],
                zone['alert_level'],
                zone['recommended_action']
            ))
        
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Stored {len(stress_zones)} stress zones")
    
    def run_prediction(self):
        """Run complete stress zone prediction"""
        logger.info("=" * 80)
        logger.info("⚠️  STRESS ZONE PREDICTION")
        logger.info("=" * 80)
        
        try:
            # Get noise predictions
            predictions_df = self.get_recent_predictions()
            
            if len(predictions_df) == 0:
                logger.warning("⚠️  No noise predictions available")
                logger.warning("   Make sure noise-predictor ran successfully")
                return
            
            logger.info(f"📊 Processing {len(predictions_df)} predictions")
            
            # Predict stress zones
            stress_zones = self.predict_stress_zones(predictions_df)
            
            # Store stress zones
            self.store_stress_zones(stress_zones)
            
            # Summary
            logger.info("=" * 80)
            logger.info("✅ STRESS ZONE PREDICTION COMPLETE!")
            logger.info("=" * 80)
            
            if stress_zones:
                alert_counts = {}
                for zone in stress_zones:
                    level = zone['alert_level']
                    alert_counts[level] = alert_counts.get(level, 0) + 1
                
                logger.info(f"\n📊 Alert Level Summary:")
                logger.info(f"   Critical: {alert_counts.get('critical', 0)}")
                logger.info(f"   High: {alert_counts.get('high', 0)}")
                logger.info(f"   Moderate: {alert_counts.get('moderate', 0)}")
                logger.info(f"   Low: {alert_counts.get('low', 0)}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Stress prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("🔌 Database connection closed")


def main():
    """Main function - runs every 15 minutes"""
    
    # Database configuration
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'traffic_noise_db'),
        'user': os.getenv('DB_USER', 'traffic_user'),
        'password': os.getenv('DB_PASSWORD', 'traffic_pass')
    }
    
    # Run interval (should match or be slightly after noise predictor)
    RUN_INTERVAL = int(os.getenv('RUN_INTERVAL', 900))  # 15 minutes
    DELAY_START = int(os.getenv('DELAY_START', 60))  # Wait 1 min for noise predictor
    
    logger.info("=" * 80)
    logger.info("🇪🇹 Dire Dawa Traffic Noise - Stress Zone Prediction Service")
    logger.info("=" * 80)
    logger.info(f"⏱️  Run Interval: Every {RUN_INTERVAL/60} minutes")
    logger.info(f"⏳ Startup Delay: {DELAY_START} seconds (wait for noise predictor)")
    logger.info(f"🗄️  Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info("=" * 80)
    
    # Create predictor
    predictor = StressZonePredictor(DB_CONFIG)
    
    if not predictor.connect():
        logger.error("❌ Failed to connect to database. Exiting.")
        sys.exit(1)
    
    # Try to load sentiment analyzer (optional)
    predictor.load_sentiment_analyzer()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run once and exit
            logger.info("🎯 Running single prediction...")
            predictor.run_prediction()
        else:
            # Initial delay
            if DELAY_START > 0:
                logger.info(f"⏳ Waiting {DELAY_START}s for noise predictor to finish...")
                time.sleep(DELAY_START)
            
            # Run continuously
            logger.info("🔄 Starting continuous prediction...")
            logger.info("   Press Ctrl+C to stop")
            
            while True:
                predictor.run_prediction()
                
                logger.info(f"\n⏸️  Waiting {RUN_INTERVAL/60} minutes until next prediction...")
                time.sleep(RUN_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping stress zone prediction (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.close()
        logger.info("✅ Stress zone prediction service stopped")


if __name__ == "__main__":
    main()
