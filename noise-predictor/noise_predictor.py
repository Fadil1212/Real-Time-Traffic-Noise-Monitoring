#!/usr/bin/env python3
"""
Noise Prediction Service
Uses trained Random Forest model to predict future noise levels
Runs every 15 minutes to generate 1-24 hour forecasts
"""

import pandas as pd
import numpy as np
import psycopg2
import joblib
import logging
from datetime import datetime, timedelta
import time
import sys
import os
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NoisePredictor:
    """Predicts future noise levels using Random Forest model"""
    
    def __init__(self, db_config, model_path):
        self.db_config = db_config
        self.model_path = model_path
        self.model = None
        self.feature_info = None
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    def load_model(self):
        """Load trained Random Forest model"""
        try:
            # Find model file
            model_pattern = os.path.join(self.model_path, 'noise_predictor_rf_*.pkl')
            model_files = glob.glob(model_pattern)
            
            if not model_files:
                logger.error(f"❌ No model found in {self.model_path}")
                logger.error(f"   Looking for pattern: {model_pattern}")
                return False
            
            # Use latest model
            latest_model = sorted(model_files)[-1]
            
            # USE JOBLIB INSTEAD OF PICKLE
            self.model = joblib.load(latest_model)  # ← CHANGED THIS LINE
            
            logger.info(f"✅ Loaded model: {os.path.basename(latest_model)}")
            logger.info(f"✅ Model type: {type(self.model).__name__}")
            logger.info(f"✅ Has predict: {hasattr(self.model, 'predict')}")
            
            # Load feature info if available
            feature_pattern = os.path.join(self.model_path, 'feature_info_*.pkl')
            feature_files = glob.glob(feature_pattern)
            
            if feature_files:
                latest_feature = sorted(feature_files)[-1]
                self.feature_info = joblib.load(latest_feature)  # ← ALSO USE JOBLIB HERE
                logger.info(f"✅ Loaded feature info: {os.path.basename(latest_feature)}")
                logger.info(f"   Features: {self.feature_info.get('features', 'N/A')}")
            else:
                logger.warning("⚠️  No feature info found - using defaults")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_recent_data(self, hours=48):
        """Get recent noise data for feature engineering"""
        query = f"""
        SELECT 
            timestamp,
            sensor_id,
            street_name,
            neighborhood,
            latitude,
            longitude,
            noise_level,
            EXTRACT(HOUR FROM timestamp) as hour,
            EXTRACT(DOW FROM timestamp) as day_of_week,
            EXTRACT(MONTH FROM timestamp) as month
        FROM noise_readings
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp DESC
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Loaded {len(df)} historical records")
        
        return df
    
    def get_locations(self):
        """Get unique locations to predict for"""
        query = """
        SELECT DISTINCT 
            street_name,
            neighborhood,
            AVG(latitude) as latitude,
            AVG(longitude) as longitude
        FROM noise_readings
        WHERE timestamp > NOW() - INTERVAL '24 hours'
        GROUP BY street_name, neighborhood
        """
        
        df = pd.read_sql(query, self.conn)
        logger.info(f"✅ Found {len(df)} locations to predict")
        
        return df
    
    def engineer_features(self, df, forecast_time):
        """
        Create features for prediction matching training
        
        Args:
            df: Historical data for this location
            forecast_time: Time to predict for
        """
        features = {}
        
        # Time-based features (matching training)
        features['hour'] = forecast_time.hour
        features['day_of_week'] = forecast_time.weekday()
        features['is_weekend'] = 1 if forecast_time.weekday() >= 5 else 0
        features['is_rush_hour'] = 1 if forecast_time.hour in [7, 8, 9, 17, 18, 19] else 0
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
        
        # Location encoding - use most common sensor/neighborhood from location data
        if len(df) > 0:
            # Get the most common sensor_id for this location
            most_common_sensor = df['sensor_id'].mode()[0] if 'sensor_id' in df.columns else 'SENSOR_DIRE_DAWA_001'
            most_common_neighborhood = df['neighborhood'].mode()[0] if 'neighborhood' in df.columns else 'Unknown'
            
            # Load mappings from feature_info if available
            if self.feature_info and 'sensor_mapping' in self.feature_info:
                sensor_mapping = {v: k for k, v in self.feature_info['sensor_mapping'].items()}
                neighborhood_mapping = {v: k for k, v in self.feature_info['neighborhood_mapping'].items()}
                
                features['sensor_encoded'] = sensor_mapping.get(most_common_sensor, 0)
                features['neighborhood_encoded'] = neighborhood_mapping.get(most_common_neighborhood, 0)
            else:
                # Fallback: use simple encoding
                features['sensor_encoded'] = 0
                features['neighborhood_encoded'] = 0
        else:
            features['sensor_encoded'] = 0
            features['neighborhood_encoded'] = 0
        
        return features
    
    def predict_location(self, location_data, historical_df, forecast_hours=[1, 3, 6, 12, 24]):
        """
        Predict noise for specific location at multiple time horizons
        
        Args:
            location_data: Dict with street_name, neighborhood, lat, lon
            historical_df: All historical data
            forecast_hours: List of hours ahead to predict
        
        Returns:
            List of prediction dicts
        """
        from datetime import timezone
        
        predictions = []
        
        # Filter for this location
        location_df = historical_df[
            (historical_df['street_name'] == location_data['street_name']) &
            (historical_df['neighborhood'] == location_data['neighborhood'])
        ].copy()
        
        if len(location_df) < 10:
            logger.warning(f"⚠️  Limited data for {location_data['street_name']} ({len(location_df)} records), using neighborhood data")
            location_df = historical_df[historical_df['neighborhood'] == location_data['neighborhood']].copy()
        
        if len(location_df) < 5:
            logger.warning(f"⚠️  Very limited data, using all data for prediction")
            location_df = historical_df.copy()
        
        # Make timezone-aware datetime
        now = datetime.now(timezone.utc)
        
        for hours_ahead in forecast_hours:
            forecast_time = now + timedelta(hours=hours_ahead)
            
            try:
                # Engineer features
                features = self.engineer_features(location_df, forecast_time)
                
                # Convert to DataFrame with correct column order
                feature_df = pd.DataFrame([features])
                
                # Ensure features are in the same order as training
                expected_features = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 
                                'hour_sin', 'hour_cos', 'sensor_encoded', 'neighborhood_encoded']
                feature_df = feature_df[expected_features]
                
                # Make prediction
                predicted_noise = self.model.predict(feature_df)[0]
                
                # Calculate confidence interval (±5 dB)
                confidence_lower = max(40.0, predicted_noise - 5.0)
                confidence_upper = min(110.0, predicted_noise + 5.0)
                
                prediction = {
                    'prediction_timestamp': now,
                    'forecast_timestamp': forecast_time,
                    'forecast_horizon': hours_ahead,
                    'street_name': location_data['street_name'],
                    'neighborhood': location_data['neighborhood'],
                    'latitude': location_data['latitude'],
                    'longitude': location_data['longitude'],
                    'predicted_noise_level': round(float(predicted_noise), 2),
                    'prediction_interval_lower': round(confidence_lower, 2),
                    'prediction_interval_upper': round(confidence_upper, 2),
                    'confidence_score': 0.85,
                    'model_name': 'RandomForest',
                    'model_version': 'v1.0'
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"❌ Prediction failed for {location_data['street_name']} at +{hours_ahead}h: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return predictions
        
    def store_predictions(self, predictions):
        """Store predictions in database"""
        if not predictions:
            logger.warning("⚠️  No predictions to store")
            return
        
        logger.info(f"💾 Storing {len(predictions)} predictions...")
        
        cursor = self.conn.cursor()
        
        for pred in predictions:
            query = """
            INSERT INTO noise_predictions (
                prediction_timestamp,
                forecast_timestamp,
                street_name,
                neighborhood,
                latitude,
                longitude,
                predicted_noise_level,
                prediction_interval_lower,
                prediction_interval_upper,
                model_name,
                model_version,
                confidence_score,
                forecast_horizon
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            
            cursor.execute(query, (
                pred['prediction_timestamp'],
                pred['forecast_timestamp'],
                pred['street_name'],
                pred['neighborhood'],
                pred['latitude'],
                pred['longitude'],
                pred['predicted_noise_level'],
                pred['prediction_interval_lower'],
                pred['prediction_interval_upper'],
                pred['model_name'],
                pred['model_version'],
                pred['confidence_score'],
                pred['forecast_horizon']
            ))
        
        self.conn.commit()
        cursor.close()
        
        logger.info(f"✅ Stored {len(predictions)} predictions")
    
    def run_prediction(self):
        """Run complete prediction cycle"""
        logger.info("=" * 80)
        logger.info("🔮 NOISE PREDICTION")
        logger.info("=" * 80)
        
        try:
            # Get historical data
            historical_df = self.get_recent_data(hours=48)
            
            if len(historical_df) < 10:
                logger.warning("⚠️  Not enough historical data (need 10+ samples)")
                return
            
            # Get locations
            locations = self.get_locations()
            
            if len(locations) == 0:
                logger.warning("⚠️  No locations found")
                return
            
            logger.info(f"📍 Predicting for {len(locations)} locations")
            
            # Predict for each location
            all_predictions = []
            
            for idx, location in locations.iterrows():
                logger.info(f"\n📊 Location {idx+1}/{len(locations)}: {location['street_name']}, {location['neighborhood']}")
                
                predictions = self.predict_location(
                    location.to_dict(),
                    historical_df,
                    forecast_hours=[1, 3, 6, 12, 24]
                )
                
                all_predictions.extend(predictions)
                
                # Show sample prediction
                if predictions:
                    sample = predictions[0]  # 1 hour ahead
                    logger.info(f"   Next hour: {sample['predicted_noise_level']:.1f} dB "
                              f"({sample['prediction_interval_lower']:.1f} - {sample['prediction_interval_upper']:.1f})")
            
            # Store all predictions
            self.store_predictions(all_predictions)
            
            logger.info("=" * 80)
            logger.info("✅ NOISE PREDICTION COMPLETE!")
            logger.info(f"   Total Predictions: {len(all_predictions)}")
            logger.info(f"   Locations: {len(locations)}")
            logger.info(f"   Time Horizons: 1h, 3h, 6h, 12h, 24h")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
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
    
    # Model path
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/rf_model')
    
    # Run interval
    RUN_INTERVAL = int(os.getenv('RUN_INTERVAL', 900))  # 15 minutes
    
    logger.info("=" * 80)
    logger.info("Melwakii Traffic Noise - Prediction Service")
    logger.info("=" * 80)
    logger.info(f"⏱️  Run Interval: Every {RUN_INTERVAL/60} minutes")
    logger.info(f"🤖 Model Path: {MODEL_PATH}")
    logger.info(f"🗄️  Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    logger.info("=" * 80)
    
    # Create predictor
    predictor = NoisePredictor(DB_CONFIG, MODEL_PATH)
    
    if not predictor.connect():
        logger.error("❌ Failed to connect to database. Exiting.")
        sys.exit(1)
    
    if not predictor.load_model():
        logger.error("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run once and exit
            logger.info("🎯 Running single prediction...")
            predictor.run_prediction()
        else:
            # Run continuously
            logger.info("🔄 Starting continuous prediction...")
            logger.info("   Press Ctrl+C to stop")
            
            while True:
                predictor.run_prediction()
                
                logger.info(f"\n⏸️  Waiting {RUN_INTERVAL/60} minutes until next prediction...")
                time.sleep(RUN_INTERVAL)
                
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping prediction service (Ctrl+C pressed)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.close()
        logger.info("✅ Prediction service stopped")


if __name__ == "__main__":
    main()