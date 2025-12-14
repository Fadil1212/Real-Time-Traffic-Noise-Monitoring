-- ============================================================================
-- Traffic Noise Monitoring System - Database Initialization Script
-- Dire Dawa, Ethiopia - Complete Schema
-- ============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================================
-- 1. NOISE ANALYSIS TABLES
-- ============================================================================

-- Raw traffic noise readings from sensors
CREATE TABLE IF NOT EXISTS noise_readings (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    noise_level DECIMAL(5, 2) NOT NULL,  -- in decibels (dB)
    sentiment_text TEXT,  -- Synthetic sentiment text based on noise level
    metadata JSONB,  -- Additional sensor metadata
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('noise_readings', 'timestamp', if_not_exists => TRUE);

-- Aggregated noise data by street/neighborhood (hourly)
CREATE TABLE IF NOT EXISTS noise_aggregates (
    timestamp TIMESTAMPTZ NOT NULL,
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    avg_noise_level DECIMAL(5, 2),
    max_noise_level DECIMAL(5, 2),
    min_noise_level DECIMAL(5, 2),
    stddev_noise_level DECIMAL(5, 2),
    reading_count INTEGER,
    is_peak_hour BOOLEAN DEFAULT FALSE,
    is_hotspot BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (timestamp, street_name)
);

SELECT create_hypertable('noise_aggregates', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- 1B. ROUTE HEAT MAP TABLE (NEW - For route-based visualization)
-- ============================================================================

-- Street segments with noise levels (for heat map routes)
CREATE TABLE IF NOT EXISTS route_noise_segments (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    street_name VARCHAR(255) NOT NULL,
    neighborhood VARCHAR(100),
    
    -- Route segment coordinates (line between two points)
    segment_start_lat DECIMAL(10, 8),
    segment_start_lon DECIMAL(11, 8),
    segment_end_lat DECIMAL(10, 8),
    segment_end_lon DECIMAL(11, 8),
    
    -- Center point of segment (for queries)
    center_lat DECIMAL(10, 8),
    center_lon DECIMAL(11, 8),
    
    -- Noise statistics for this segment
    avg_noise_level DECIMAL(5, 2),
    max_noise_level DECIMAL(5, 2),
    min_noise_level DECIMAL(5, 2),
    noise_category VARCHAR(20), -- 'low', 'moderate', 'high', 'critical'
    
    -- Segment properties
    segment_length_meters DECIMAL(10, 2),
    bearing DECIMAL(5, 2), -- Direction in degrees (0-360)
    
    -- Analytics
    reading_count INTEGER,
    sensors_in_segment TEXT, -- Comma-separated sensor IDs
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    -- Composite primary key including timestamp for TimescaleDB
    PRIMARY KEY (id, timestamp)
);

-- Indexes for route segments
CREATE INDEX IF NOT EXISTS idx_route_segments_street ON route_noise_segments(street_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_segments_neighborhood ON route_noise_segments(neighborhood, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_route_segments_noise ON route_noise_segments(noise_category, avg_noise_level DESC);
CREATE INDEX IF NOT EXISTS idx_route_segments_timestamp ON route_noise_segments(timestamp DESC);

SELECT create_hypertable('route_noise_segments', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- 2. SENTIMENT ANALYSIS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    location_name VARCHAR(255),
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    sentiment_score DECIMAL(3, 2),
    sentiment_label VARCHAR(20),
    confidence DECIMAL(3, 2),
    text_source VARCHAR(50),
    keywords TEXT[],
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('sentiment_scores', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- 3. CORRELATION ANALYSIS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS noise_sentiment_correlation (
    id SERIAL,
    analysis_timestamp TIMESTAMPTZ NOT NULL,
    time_window VARCHAR(50),
    location_name VARCHAR(255),
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    avg_noise_level DECIMAL(5, 2),
    avg_sentiment_score DECIMAL(3, 2),
    correlation_coefficient DECIMAL(5, 4),
    sample_size INTEGER,
    statistical_significance DECIMAL(5, 4),
    PRIMARY KEY (id, analysis_timestamp)
);

SELECT create_hypertable('noise_sentiment_correlation', 'analysis_timestamp', if_not_exists => TRUE);

-- ============================================================================
-- 4. PREDICTIVE ANALYSIS TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS noise_predictions (
    id SERIAL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    predicted_noise_level DECIMAL(5, 2),
    prediction_interval_lower DECIMAL(5, 2),
    prediction_interval_upper DECIMAL(5, 2),
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    confidence_score DECIMAL(3, 2),
    forecast_horizon INTEGER,
    PRIMARY KEY (id, prediction_timestamp)
);

SELECT create_hypertable('noise_predictions', 'prediction_timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS predicted_stress_zones (
    id SERIAL,
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    forecast_timestamp TIMESTAMPTZ NOT NULL,
    zone_name VARCHAR(255),
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    predicted_stress_level DECIMAL(5, 2),
    predicted_noise_contribution DECIMAL(5, 2),
    predicted_sentiment_contribution DECIMAL(5, 2),
    alert_level VARCHAR(20),
    recommended_action TEXT,
    PRIMARY KEY (id, prediction_timestamp)
);

SELECT create_hypertable('predicted_stress_zones', 'prediction_timestamp', if_not_exists => TRUE);

-- ============================================================================
-- 5. STRESS/HOTSPOT IDENTIFICATION TABLES
-- ============================================================================

CREATE TABLE IF NOT EXISTS current_stress_zones (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    zone_name VARCHAR(255),
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    stress_score DECIMAL(5, 2),
    noise_contribution DECIMAL(5, 2),
    sentiment_contribution DECIMAL(5, 2),
    current_noise_level DECIMAL(5, 2),
    current_sentiment_score DECIMAL(3, 2),
    alert_level VARCHAR(20),
    is_recurring BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('current_stress_zones', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS peak_noise_patterns (
    id SERIAL PRIMARY KEY,
    location_name VARCHAR(255),
    street_name VARCHAR(255),
    neighborhood VARCHAR(100),
    day_of_week INTEGER,
    hour_of_day INTEGER,
    avg_noise_level DECIMAL(5, 2),
    max_noise_level DECIMAL(5, 2),
    frequency_count INTEGER,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 6. ALTERNATIVE ROUTES / RECOMMENDATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS alternative_routes (
    id SERIAL PRIMARY KEY,
    route_name VARCHAR(255),
    origin_street VARCHAR(255),
    destination_street VARCHAR(255),
    route_points JSONB,
    avg_noise_level DECIMAL(5, 2),
    estimated_duration_minutes INTEGER,
    noise_reduction_benefit DECIMAL(5, 2),
    is_recommended BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS route_recommendations (
    id SERIAL,
    timestamp TIMESTAMPTZ NOT NULL,
    user_origin VARCHAR(255),
    user_destination VARCHAR(255),
    recommended_route_id INTEGER REFERENCES alternative_routes(id),
    current_noise_level DECIMAL(5, 2),
    route_noise_level DECIMAL(5, 2),
    reason TEXT,
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('route_recommendations', 'timestamp', if_not_exists => TRUE);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_noise_readings_sensor ON noise_readings(sensor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_noise_readings_street ON noise_readings(street_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_noise_readings_neighborhood ON noise_readings(neighborhood, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_noise_readings_level ON noise_readings(noise_level DESC);

CREATE INDEX IF NOT EXISTS idx_noise_aggregates_street ON noise_aggregates(street_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_noise_aggregates_hotspot ON noise_aggregates(is_hotspot, timestamp DESC) WHERE is_hotspot = TRUE;
CREATE INDEX IF NOT EXISTS idx_noise_aggregates_peak ON noise_aggregates(is_peak_hour, timestamp DESC) WHERE is_peak_hour = TRUE;

CREATE INDEX IF NOT EXISTS idx_sentiment_location ON sentiment_scores(location_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_label ON sentiment_scores(sentiment_label, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_correlation_location ON noise_sentiment_correlation(location_name, analysis_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_forecast ON noise_predictions(forecast_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON noise_predictions(model_name, prediction_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_stress_alert_level ON current_stress_zones(alert_level, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_stress_recurring ON current_stress_zones(is_recurring, timestamp DESC) WHERE is_recurring = TRUE;

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS noise_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    street_name,
    neighborhood,
    AVG(latitude) as avg_latitude,
    AVG(longitude) as avg_longitude,
    AVG(noise_level) as avg_noise,
    MAX(noise_level) as max_noise,
    MIN(noise_level) as min_noise,
    STDDEV(noise_level) as stddev_noise,
    COUNT(*) as reading_count
FROM noise_readings
GROUP BY hour, street_name, neighborhood
WITH NO DATA;

SELECT add_continuous_aggregate_policy('noise_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes',
    if_not_exists => TRUE
);

CREATE MATERIALIZED VIEW IF NOT EXISTS noise_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    street_name,
    neighborhood,
    AVG(noise_level) as avg_noise,
    MAX(noise_level) as max_noise,
    MIN(noise_level) as min_noise,
    COUNT(*) as reading_count
FROM noise_readings
GROUP BY day, street_name, neighborhood
WITH NO DATA;

SELECT add_continuous_aggregate_policy('noise_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Stress score calculation
CREATE OR REPLACE FUNCTION calculate_stress_score(
    noise_level DECIMAL,
    sentiment_score DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    RETURN (
        (0.7 * ((noise_level - 40) / 60 * 100)) +
        (0.3 * ((1 - sentiment_score) / 2 * 100))
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Alert level determination
CREATE OR REPLACE FUNCTION get_alert_level(stress_score DECIMAL) 
RETURNS VARCHAR AS $$
BEGIN
    IF stress_score >= 85 THEN RETURN 'critical';
    ELSIF stress_score >= 70 THEN RETURN 'high';
    ELSIF stress_score >= 50 THEN RETURN 'moderate';
    ELSE RETURN 'low';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Noise category for route segments
CREATE OR REPLACE FUNCTION get_noise_category(noise_level DECIMAL) 
RETURNS VARCHAR AS $$
BEGIN
    IF noise_level >= 90 THEN RETURN 'critical';
    ELSIF noise_level >= 80 THEN RETURN 'high';
    ELSIF noise_level >= 70 THEN RETURN 'moderate';
    ELSE RETURN 'low';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Haversine distance (meters)
CREATE OR REPLACE FUNCTION calculate_distance(
    lat1 DECIMAL, lon1 DECIMAL, 
    lat2 DECIMAL, lon2 DECIMAL
) RETURNS DECIMAL AS $$
DECLARE
    earth_radius CONSTANT DECIMAL := 6371000;
    dlat DECIMAL;
    dlon DECIMAL;
    a DECIMAL;
    c DECIMAL;
BEGIN
    dlat := radians(lat2 - lat1);
    dlon := radians(lon2 - lon1);
    
    a := sin(dlat/2) * sin(dlat/2) + 
         cos(radians(lat1)) * cos(radians(lat2)) * 
         sin(dlon/2) * sin(dlon/2);
    c := 2 * atan2(sqrt(a), sqrt(1-a));
    
    RETURN earth_radius * c;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Bearing calculation (degrees)
CREATE OR REPLACE FUNCTION calculate_bearing(
    lat1 DECIMAL, lon1 DECIMAL,
    lat2 DECIMAL, lon2 DECIMAL
) RETURNS DECIMAL AS $$
DECLARE
    dlon DECIMAL;
    y DECIMAL;
    x DECIMAL;
    bearing DECIMAL;
BEGIN
    dlon := radians(lon2 - lon1);
    
    y := sin(dlon) * cos(radians(lat2));
    x := cos(radians(lat1)) * sin(radians(lat2)) - 
         sin(radians(lat1)) * cos(radians(lat2)) * cos(dlon);
    
    bearing := degrees(atan2(y, x));
    
    RETURN MOD(bearing + 360, 360);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO traffic_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO traffic_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO traffic_user;

-- ============================================================================
-- COMPLETION
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '  Traffic Noise Monitoring Database - Initialized Successfully!';
    RAISE NOTICE '  Dire Dawa, Ethiopia';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Created Tables (12 total):';
    RAISE NOTICE '  1. noise_readings';
    RAISE NOTICE '  2. noise_aggregates';
    RAISE NOTICE '  3. route_noise_segments (NEW - for heat map)';
    RAISE NOTICE '  4. sentiment_scores';
    RAISE NOTICE '  5. noise_sentiment_correlation';
    RAISE NOTICE '  6. noise_predictions';
    RAISE NOTICE '  7. predicted_stress_zones';
    RAISE NOTICE '  8. current_stress_zones';
    RAISE NOTICE '  9. peak_noise_patterns';
    RAISE NOTICE ' 10. alternative_routes';
    RAISE NOTICE ' 11. route_recommendations';
    RAISE NOTICE ' 12. Continuous aggregates (noise_hourly, noise_daily)';
    RAISE NOTICE '';
    RAISE NOTICE 'Helper Functions:';
    RAISE NOTICE '  ✓ calculate_stress_score()';
    RAISE NOTICE '  ✓ get_alert_level()';
    RAISE NOTICE '  ✓ get_noise_category()';
    RAISE NOTICE '  ✓ calculate_distance()';
    RAISE NOTICE '  ✓ calculate_bearing()';
    RAISE NOTICE '';
    RAISE NOTICE '========================================================================';
END $$;