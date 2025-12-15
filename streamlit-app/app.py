"""
Milwaukee Traffic Noise Monitoring Dashboard - COMPLETE EDITION
Professional industry-standard implementation with ALL features:
- Route Heat Maps
- Noise Predictions
- Stress Zone Detection
- Correlation Analysis
- Route Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import psycopg2
from datetime import datetime, timedelta
import logging
import requests
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Milwaukee Noise Monitoring",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .critical-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .high-alert {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 5px 0;
    }
    .correlation-positive {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
    }
    .correlation-negative {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def display_large_dataframe(df, title="Data", default_rows=100, max_rows=500):
    '''
    Smart display for large dataframes with filters and download
    
    Args:
        df: DataFrame to display
        title: Table title
        default_rows: Default number of rows to show
        max_rows: Maximum rows user can select
    '''
    if df.empty:
        st.info(f"No {title.lower()} available")
        return
    
    st.markdown(f"### {title}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        show_count = st.slider(
            f"Rows to display",
            10,
            min(max_rows, len(df)),
            min(default_rows, len(df)),
            step=10,
            key=f'slider_{title}'
        )
    
    with col2:
        st.metric("Total Rows", f"{len(df):,}")
    
    st.caption(f"Showing {show_count} of {len(df):,} rows")
    
    # Display
    st.dataframe(df.head(show_count), use_container_width=True, height=400)
    
    # Download
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download Full {title}",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f'download_{title}'
    )

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Create cached database connection"""
    try:
        conn = psycopg2.connect(
            host=st.secrets.get("POSTGRES_HOST", "postgres"),
            port=st.secrets.get("POSTGRES_PORT", 5432),
            database=st.secrets.get("POSTGRES_DB", "traffic_noise_db"),
            user=st.secrets.get("POSTGRES_USER", "traffic_user"),
            password=st.secrets.get("POSTGRES_PASSWORD", "traffic_pass")
        )
        logger.info("✅ Database connected")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        st.error(f"Database connection error: {e}")
        return None

# ============================================================================
# DATA FETCHING - EXISTING
# ============================================================================

@st.cache_data(ttl=10)
def fetch_latest_readings(hours=1):
    """Fetch recent noise readings"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = f"""
    SELECT 
        timestamp,
        sensor_id,
        street_name,
        neighborhood,
        latitude,
        longitude,
        noise_level
    FROM noise_readings
    WHERE timestamp > NOW() - INTERVAL '{hours} hours'
    ORDER BY timestamp DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"📊 Fetched {len(df)} noise readings")
        return df
    except Exception as e:
        logger.error(f"Error fetching readings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_route_segments(hours=2):
    """Fetch route heat map segments"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = f"""
    SELECT 
        timestamp,
        street_name,
        neighborhood,
        segment_start_lat,
        segment_start_lon,
        segment_end_lat,
        segment_end_lon,
        center_lat,
        center_lon,
        avg_noise_level,
        max_noise_level,
        min_noise_level,
        noise_category,
        segment_length_meters,
        sensors_in_segment
    FROM route_noise_segments
    WHERE timestamp > NOW() - INTERVAL '{hours} hours'
    ORDER BY avg_noise_level DESC
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"🗺️  Fetched {len(df)} route segments")
        return df
    except Exception as e:
        logger.error(f"Error fetching routes: {e}")
        return pd.DataFrame()

# ============================================================================
# DATA FETCHING - NEW FEATURES
# ============================================================================

@st.cache_data(ttl=300)
def fetch_noise_predictions(hours_ahead=24):
    """Fetch noise predictions"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = f"""
    SELECT 
        prediction_timestamp,
        forecast_timestamp,
        street_name,
        neighborhood,
        latitude,
        longitude,
        predicted_noise_level,
        prediction_interval_lower,
        prediction_interval_upper,
        confidence_score,
        forecast_horizon
    FROM noise_predictions
    WHERE forecast_timestamp BETWEEN NOW() AND NOW() + INTERVAL '{hours_ahead} hours'
    ORDER BY prediction_timestamp DESC, forecast_horizon
    LIMIT 1000
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"🔮 Fetched {len(df)} predictions")
        return df
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_stress_zones():
    """Fetch predicted stress zones"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
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
    FROM predicted_stress_zones
    WHERE forecast_timestamp > NOW()
    ORDER BY predicted_stress_level DESC
    LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"😰 Fetched {len(df)} stress zones")
        return df
    except Exception as e:
        logger.error(f"Error fetching stress zones: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_correlations():
    """Fetch noise-sentiment correlations"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        analysis_timestamp,
        time_window,
        location_name,
        street_name,
        neighborhood,
        latitude,
        longitude,
        avg_noise_level,
        avg_sentiment_score,
        correlation_coefficient,
        sample_size,
        statistical_significance
    FROM noise_sentiment_correlation
    ORDER BY analysis_timestamp DESC
    LIMIT 100
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"📊 Fetched {len(df)} correlations")
        return df
    except Exception as e:
        logger.error(f"Error fetching correlations: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_route_recommendations():
    """Fetch route recommendations"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
    SELECT 
        rr.timestamp,
        rr.user_origin,
        rr.user_destination,
        rr.current_noise_level,
        rr.route_noise_level,
        rr.reason,
        ar.route_name,
        ar.route_points,
        ar.avg_noise_level,
        ar.estimated_duration_minutes,
        ar.noise_reduction_benefit
    FROM route_recommendations rr
    LEFT JOIN alternative_routes ar ON rr.recommended_route_id = ar.id
    ORDER BY rr.timestamp DESC
    LIMIT 50
    """
    
    try:
        df = pd.read_sql(query, conn)
        logger.info(f"🗺️  Fetched {len(df)} route recommendations")
        return df
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_statistics():
    """Fetch summary statistics"""
    conn = get_db_connection()
    if not conn:
        return {}
    
    query = """
    SELECT 
        COUNT(*) as total_readings,
        COUNT(DISTINCT sensor_id) as active_sensors,
        ROUND(AVG(noise_level)::numeric, 1) as avg_noise,
        ROUND(MAX(noise_level)::numeric, 1) as max_noise,
        ROUND(MIN(noise_level)::numeric, 1) as min_noise,
        COUNT(*) FILTER (WHERE noise_level >= 90) as critical_count,
        COUNT(*) FILTER (WHERE noise_level >= 80 AND noise_level < 90) as high_count
    FROM noise_readings
    WHERE timestamp > NOW() - INTERVAL '1 hour'
    """
    
    try:
        df = pd.read_sql(query, conn)
        stats = df.iloc[0].to_dict() if not df.empty else {}
        logger.info(f"📊 Fetched statistics")
        return stats
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return {}

# ============================================================================
# ROUTE RECOMMENDATION API INTEGRATION
# ============================================================================

def render_route_recommendations_tab(df_recommendations):
    """
    FIX #4: OPTIMIZED Route Recommendations Tab
    
    Replace your existing Tab 5 content with this
    """
    st.subheader("🛣️ AI-Powered Route Recommendations")
    
    # Check API health
    col_health1, col_health2 = st.columns([3, 1])
    
    with col_health2:
        if st.button("🔄 Check API Status", use_container_width=True):
            try:
                response = requests.get("http://alternative-routes-api:5000/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    cache = health.get('cache', {})
                    
                    if cache.get('cached'):
                        st.success(f"✅ API Ready\n\n"
                                 f"📊 {cache.get('locations', 0)} locations cached\n\n"
                                 f"⏱️ Cache age: {cache.get('age_seconds', 0)}s")
                    else:
                        st.warning("⚠️ API initializing...\n\nGraph cache building. First request may be slow.")
                else:
                    st.error("❌ API offline")
            except:
                st.error("🔌 Cannot reach API")
    
    # Initialize session state for storing route results
    if 'last_route_result' not in st.session_state:
        st.session_state.last_route_result = None
    if 'last_route_time' not in st.session_state:
        st.session_state.last_route_time = None
    
    # Route calculator
    st.markdown("### 🗺️ Find Quiet Routes")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Try to get locations from API
        locations = get_available_locations()
        
        if locations:
            location_names = sorted(list(set([loc['street'] for loc in locations])))
            origin = st.selectbox("Origin", location_names, key='origin_opt')
        else:
            origin = st.text_input("Origin", "Wisconsin Ave", key='origin_opt')
            st.caption("⚠️ Could not load location list. Enter manually.")
    
    with col2:
        if locations:
            destination = st.selectbox("Destination", location_names, key='destination_opt')
        else:
            destination = st.text_input("Destination", "Water St", key='destination_opt')
    
    with col3:
        st.write("")  # Spacing
        st.write("")
        calculate_btn = st.button("🔍 Find Routes", use_container_width=True, type="primary", key='calc_routes_opt')
    
    # Calculate routes
    if calculate_btn and origin and destination:
        # Call optimized API with infinite timeout
        result = call_route_api(origin, destination)
        
        # Store in session state
        if result:
            st.session_state.last_route_result = result
            st.session_state.last_route_time = datetime.now()
    
    # Display results from session state (persists after button click)
    result = st.session_state.last_route_result
    
    if result and result.get('success'):
        routes = result.get('routes', [])
        
        # Show when calculated
        if st.session_state.last_route_time:
            age = (datetime.now() - st.session_state.last_route_time).total_seconds()
            st.caption(f"🕐 Calculated {age:.0f} seconds ago")
        
        if routes:
            st.success(f"✅ Found {len(routes)} alternative route(s)!")
            
            # Map visualization
            st.markdown("### 🗺️ Route Visualization")
            
            route_map = create_recommended_routes_map(result)
            if route_map:
                st.pydeck_chart(route_map, use_container_width=True)
                
                # Legend
                col_legend1, col_legend2, col_legend3 = st.columns(3)
                with col_legend1:
                    st.markdown("🟢 **Quietest Route** - Lowest noise levels")
                with col_legend2:
                    st.markdown("🔵 **Shortest Route** - Minimum distance")
                with col_legend3:
                    st.markdown("🟡 **Balanced Route** - Optimized noise & distance")
            else:
                st.info("""
                ℹ️ **Map visualization temporarily unavailable**
                
                The system is still collecting street coordinate data. The route calculation is working correctly,
                but we need more historical data to display the map. Please check back in a few minutes, or see
                the detailed route information below.
                """)
            
            st.markdown("---")
            st.markdown("### 📊 Route Details")
            
            for route in routes:
                rank = route['rank']
                route_type = route['type']
                
                # Color based on noise
                noise_color = get_noise_color_hex(route['avg_noise_level'])
                
                with st.expander(f"🏆 Route {rank}: {route_type.upper()} ({route['avg_noise_level']:.1f} dB)", expanded=(rank==1)):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Distance", f"{route['distance_meters']:.0f}m")
                    
                    with col2:
                        st.metric("Time", f"~{route['estimated_time_minutes']:.1f} min")
                    
                    with col3:
                        st.metric("Noise Level", f"{route['avg_noise_level']:.1f} dB")
                    
                    with col4:
                        st.markdown(
                            f'<div style="background-color:{noise_color};padding:10px;border-radius:5px;text-align:center;color:white;">'
                            f'<b>{route["noise_category"].upper()}</b></div>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown(f"**Path:** {' → '.join(route['path'][:10])}")
                    if len(route['path']) > 10:
                        st.caption(f"...and {len(route['path']) - 10} more segments")
                    
                    st.info(f"💡 {route['recommendation']}")
        else:
            st.error("❌ No routes found between these locations")
            
    elif result and not result.get('success'):
        # Show detailed error with troubleshooting
        error_msg = result.get('user_message', result.get('error', 'Unknown error'))
        st.error(f"❌ {error_msg}")
        
        # Show troubleshooting if available
        if result.get('troubleshooting'):
            with st.expander("🔧 Troubleshooting Tips"):
                for tip in result['troubleshooting']:
                    st.write(f"• {tip}")
        
        # Show available locations if provided
        if result.get('available_locations'):
            with st.expander("📍 Available Locations (sample)"):
                for loc in result['available_locations'][:20]:
                    st.write(f"• {loc}")
    
    st.markdown("---")
    
    # Recent recommendations
    st.markdown("### 📜 Recent Route Recommendations")
    
    if df_recommendations.empty:
        st.info("No recent route recommendations")
    else:
        recent_recs = df_recommendations.head(10).copy()
        
        for idx, rec in recent_recs.iterrows():
            with st.expander(f"🗺️ {rec['user_origin']} → {rec['user_destination']} ({rec['timestamp'].strftime('%H:%M')})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Route Noise", f"{rec['route_noise_level']:.1f} dB")
                
                with col2:
                    if pd.notna(rec['estimated_duration_minutes']):
                        st.metric("Duration", f"{rec['estimated_duration_minutes']:.0f} min")
                
                with col3:
                    if pd.notna(rec['noise_reduction_benefit']):
                        st.metric("Noise Reduction", f"{rec['noise_reduction_benefit']:.1f} dB")
                
                if pd.notna(rec['reason']):
                    st.info(f"💡 {rec['reason']}")

def call_route_api(origin, destination):
    """
    FIX #4: Call route API with INFINITE timeout and progress indicator
    
    Changes:
    1. timeout=None → infinite timeout
    2. Added progress bar
    3. Added status messages
    4. Better error handling
    """
    try:
        # Show progress indicator
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with progress_placeholder:
            with st.spinner('🔍 Calculating optimal routes...'):
                status_placeholder.info("⏳ Building route graph... (this may take 30-60 seconds on first request)")
                
                start_time = time.time()
                
                # FIX #4: timeout=None for infinite timeout
                response = requests.post(
                    "http://alternative-routes-api:5000/api/route-recommendations",
                    json={"origin": origin, "destination": destination},
                    timeout=None  # ← INFINITE TIMEOUT!
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Clear status messages
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    # Show success message
                    st.success(f"✅ Routes calculated in {elapsed:.1f} seconds")
                    
                    return result
                else:
                    progress_placeholder.empty()
                    status_placeholder.empty()
                    
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get('user_message', 'Unknown error')
                    st.error(f"❌ {error_msg}")
                    
                    return error_data
                    
    except requests.exceptions.Timeout:
        # This should never happen with timeout=None, but just in case
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error("⏱️ Request timed out. Try again.")
        return None
        
    except requests.exceptions.ConnectionError:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error("🔌 Cannot connect to route service. Is it running?")
        return None
        
    except Exception as e:
        progress_placeholder.empty()
        status_placeholder.empty()
        st.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def get_available_locations():
    """
    FIX #4: Get locations with timeout and caching
    """
    try:
        response = requests.get(
            "http://alternative-routes-api:5000/api/locations",
            timeout=30  # Short timeout for location list
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('locations', [])
        else:
            return []
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return []

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_noise_color(noise_level):
    """Get color for noise level (RGB)"""
    if noise_level >= 90:
        return [255, 0, 0, 200]
    elif noise_level >= 80:
        return [255, 165, 0, 200]
    elif noise_level >= 70:
        return [255, 255, 0, 200]
    else:
        return [0, 255, 0, 200]

def get_noise_color_hex(noise_level):
    """Get hex color for noise level"""
    if noise_level >= 90:
        return "#FF0000"
    elif noise_level >= 80:
        return "#FFA500"
    elif noise_level >= 70:
        return "#FFFF00"
    else:
        return "#00FF00"

def get_stress_color(stress_level):
    """Get color for stress level"""
    if stress_level >= 85:
        return [139, 0, 0, 220]  # Dark red
    elif stress_level >= 70:
        return [255, 0, 0, 200]  # Red
    elif stress_level >= 50:
        return [255, 165, 0, 200]  # Orange
    else:
        return [255, 255, 0, 200]  # Yellow

def interpolate_route_color(noise_level, min_noise=50, max_noise=100):
    """Interpolate color based on noise level for smooth gradient"""
    normalized = np.clip((noise_level - min_noise) / (max_noise - min_noise), 0, 1)
    
    if normalized < 0.25:
        r = int(normalized * 4 * 255)
        g = 255
        b = 0
    elif normalized < 0.5:
        r = 255
        g = int(255 - (normalized - 0.25) * 4 * 100)
        b = 0
    elif normalized < 0.75:
        r = 255
        g = int(155 - (normalized - 0.5) * 4 * 155)
        b = 0
    else:
        r = 255
        g = 0
        b = 0
    
    return [r, g, b, 200]

# ============================================================================
# VISUALIZATION FUNCTIONS - PREDICTIONS
# ============================================================================

def create_prediction_timeline(df_pred):
    """Create prediction timeline with confidence intervals"""
    if df_pred.empty:
        return None
    
    # Group by location
    locations = df_pred.groupby(['street_name', 'neighborhood']).size().nlargest(5).index
    
    fig = go.Figure()
    
    for street, neighborhood in locations:
        loc_data = df_pred[
            (df_pred['street_name'] == street) & 
            (df_pred['neighborhood'] == neighborhood)
        ].sort_values('forecast_timestamp')
        
        if len(loc_data) == 0:
            continue
        
        location_label = f"{street}, {neighborhood}"
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=loc_data['forecast_timestamp'],
            y=loc_data['predicted_noise_level'],
            name=location_label,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([loc_data['forecast_timestamp'], loc_data['forecast_timestamp'][::-1]]),
            y=pd.concat([loc_data['prediction_interval_upper'], loc_data['prediction_interval_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{location_label} CI'
        ))
    
    fig.update_layout(
        title='Noise Level Predictions (Next 24 Hours)',
        xaxis_title='Time',
        yaxis_title='Predicted Noise (dB)',
        template='plotly_white',
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_prediction_heatmap(df_pred):
    """Create heatmap of predictions by location and time"""
    if df_pred.empty:
        return None
    
    # Pivot data
    df_pivot = df_pred.pivot_table(
        values='predicted_noise_level',
        index='street_name',
        columns='forecast_horizon',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=[f'+{h}h' for h in df_pivot.columns],
        y=df_pivot.index,
        colorscale='RdYlGn_r',
        text=df_pivot.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(title="Noise (dB)")
    ))
    
    fig.update_layout(
        title='Predicted Noise by Location and Time Horizon',
        xaxis_title='Forecast Horizon',
        yaxis_title='Location',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_prediction_map(df_pred):
    """
    Create map of predicted noise levels
    
    FIX: PyDeck tooltips don't support Python format specifiers like :.1f or :.0%
    Solution: Pre-format numbers in the dataframe before passing to PyDeck
    """
    if df_pred.empty:
        return None
    
    # Get latest prediction for each location
    latest_pred = df_pred.sort_values('prediction_timestamp').groupby(
        ['street_name', 'neighborhood']
    ).last().reset_index()
    
    # Add color
    latest_pred['color'] = latest_pred['predicted_noise_level'].apply(get_noise_color)
    latest_pred['radius'] = 100
    
    # ============================================================================
    # FIX: Pre-format numbers for PyDeck tooltip
    # PyDeck doesn't support Python format specifiers, so we format in advance
    # ============================================================================
    latest_pred['predicted_noise_formatted'] = latest_pred['predicted_noise_level'].apply(
        lambda x: f"{x:.1f}"
    )
    latest_pred['confidence_formatted'] = latest_pred['confidence_score'].apply(
        lambda x: f"{x:.0%}"
    )
    
    # Center
    center_lat = latest_pred['latitude'].mean()
    center_lon = latest_pred['longitude'].mean()
    
    # Create layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        latest_pred,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius='radius',
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        get_line_color=[255, 255, 255],
    )
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=0,
    )
    
    # ============================================================================
    # FIX: Use pre-formatted columns in tooltip (no format specifiers)
    # ============================================================================
    tooltip = {
        "html": "<b>{street_name}</b><br/>"
                "Predicted: <b>{predicted_noise_formatted} dB</b><br/>"
                "Confidence: {confidence_formatted}<br/>"
                "Neighborhood: {neighborhood}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    )
    
    return deck

# ============================================================================
# VISUALIZATION FUNCTIONS - STRESS ZONES
# ============================================================================

def create_stress_zone_map(df_stress):
    """
    Create map of predicted stress zones
    
    FIX: PyDeck tooltips don't support Python format specifiers like :.1f
    Solution: Pre-format numbers in the dataframe before passing to PyDeck
    """
    if df_stress.empty:
        return None
    
    # Add color based on stress level
    df_stress['color'] = df_stress['predicted_stress_level'].apply(get_stress_color)
    df_stress['radius'] = df_stress['predicted_stress_level'] * 2  # Scale with stress
    
    # ============================================================================
    # FIX: Pre-format numbers for PyDeck tooltip
    # PyDeck doesn't support Python format specifiers, so we format in advance
    # ============================================================================
    df_stress['stress_level_formatted'] = df_stress['predicted_stress_level'].apply(
        lambda x: f"{x:.1f}"
    )
    df_stress['noise_contribution_formatted'] = df_stress['predicted_noise_contribution'].apply(
        lambda x: f"{x:.1f}"
    )
    df_stress['sentiment_contribution_formatted'] = df_stress['predicted_sentiment_contribution'].apply(
        lambda x: f"{x:.1f}"
    )
    
    # Center
    center_lat = df_stress['latitude'].mean()
    center_lon = df_stress['longitude'].mean()
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_stress,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius='radius',
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        line_width_min_pixels=3,
        get_line_color=[255, 255, 255],
    )
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=0,
    )
    
    # ============================================================================
    # FIX: Use pre-formatted columns in tooltip (no format specifiers)
    # ============================================================================
    tooltip = {
        "html": "<b>{zone_name}</b><br/>"
                "Stress Level: <b>{stress_level_formatted}</b><br/>"
                "Alert: <b>{alert_level}</b><br/>"
                "Noise: {noise_contribution_formatted}<br/>"
                "Sentiment: {sentiment_contribution_formatted}<br/>"
                "Action: {recommended_action}",
        "style": {
            "backgroundColor": "darkred",
            "color": "white",
            "fontSize": "12px"
        }
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    )
    
    return deck

def create_stress_breakdown(df_stress):
    """Create breakdown of stress contributions"""
    if df_stress.empty:
        return None
    
    # Average contributions by alert level
    breakdown = df_stress.groupby('alert_level').agg({
        'predicted_noise_contribution': 'mean',
        'predicted_sentiment_contribution': 'mean',
        'predicted_stress_level': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Noise Contribution',
        x=breakdown['alert_level'],
        y=breakdown['predicted_noise_contribution'],
        marker_color='#ff4444'
    ))
    
    fig.add_trace(go.Bar(
        name='Sentiment Contribution',
        x=breakdown['alert_level'],
        y=breakdown['predicted_sentiment_contribution'],
        marker_color='#4444ff'
    ))
    
    fig.update_layout(
        title='Stress Contributors by Alert Level',
        xaxis_title='Alert Level',
        yaxis_title='Contribution Score',
        barmode='stack',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_stress_gauge(stress_level, alert_level):
    """Create gauge chart for stress level"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stress_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Stress Level ({alert_level.upper()})"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if stress_level >= 85 else "orange" if stress_level >= 70 else "yellow"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "lightyellow"},
                {'range': [70, 85], 'color': "lightcoral"},
                {'range': [85, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig

# ============================================================================
# VISUALIZATION FUNCTIONS - CORRELATIONS
# ============================================================================

def create_correlation_scatter(df_corr):
    """Create scatter plot of noise vs sentiment"""
    if df_corr.empty:
        return None
    
    # Ensure sample_size is positive (should always be, but safeguard)
    df_corr_plot = df_corr.copy()
    df_corr_plot['sample_size'] = df_corr_plot['sample_size'].clip(lower=1)
    
    fig = px.scatter(
        df_corr_plot,
        x='avg_noise_level',
        y='avg_sentiment_score',
        size='sample_size',
        color='correlation_coefficient',
        hover_data=['location_name', 'time_window'],
        title='Noise-Sentiment Correlation Analysis',
        labels={
            'avg_noise_level': 'Average Noise Level (dB)',
            'avg_sentiment_score': 'Average Sentiment Score',
            'correlation_coefficient': 'Correlation'
        },
        color_continuous_scale='RdYlGn',
        template='plotly_white',
        height=500
    )
    
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1 (Negative)", "-0.5", "0 (None)", "0.5", "1 (Positive)"]
        )
    )
    
    return fig

def create_correlation_heatmap(df_corr):
    """Create heatmap of correlations by location"""
    if df_corr.empty:
        return None
    
    # Pivot by location
    df_pivot = df_corr.pivot_table(
        values='correlation_coefficient',
        index='street_name',
        columns='time_window',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=df_pivot.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Coefficient by Location and Time Window',
        xaxis_title='Time Window',
        yaxis_title='Location',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_correlation_distribution(df_corr):
    """Create distribution of correlation coefficients"""
    if df_corr.empty:
        return None
    
    fig = px.histogram(
        df_corr,
        x='correlation_coefficient',
        nbins=30,
        title='Distribution of Noise-Sentiment Correlations',
        labels={'correlation_coefficient': 'Correlation Coefficient', 'count': 'Frequency'},
        template='plotly_white',
        color_discrete_sequence=['#1f77b4'],
        height=350
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="No Correlation")
    fig.add_vline(x=-0.5, line_dash="dash", line_color="red", annotation_text="Strong Negative")
    fig.add_vline(x=0.5, line_dash="dash", line_color="green", annotation_text="Strong Positive")
    
    return fig

# ============================================================================
# VISUALIZATION FUNCTIONS - ROUTES (EXISTING)
# ============================================================================

def create_route_heatmap(df_routes):
    """Create route heat map using PyDeck"""
    if df_routes.empty:
        return None
    
    route_lines = []
    
    for idx, row in df_routes.iterrows():
        route_lines.append({
            'path': [
                [row['segment_start_lon'], row['segment_start_lat']],
                [row['segment_end_lon'], row['segment_end_lat']]
            ],
            'color': interpolate_route_color(row['avg_noise_level']),
            'width': 8,
            'street_name': row['street_name'],
            'noise': row['avg_noise_level'],
            'category': row['noise_category'],
            'neighborhood': row['neighborhood']
        })
    
    df_lines = pd.DataFrame(route_lines)
    
    all_lats = df_routes[['segment_start_lat', 'segment_end_lat']].values.flatten()
    all_lons = df_routes[['segment_start_lon', 'segment_end_lon']].values.flatten()
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    layer = pdk.Layer(
        "PathLayer",
        df_lines,
        get_path='path',
        get_color='color',
        get_width='width',
        width_scale=1,
        width_min_pixels=3,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 100],
    )
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        pitch=0,
    )
    
    tooltip = {
        "html": "<b>{street_name}</b><br/>"
                "Avg Noise: <b>{noise:.1f} dB</b><br/>"
                "Category: <b>{category}</b><br/>"
                "Neighborhood: {neighborhood}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "14px"
        }
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    )
    
    return deck

def create_route_heatmap_3d(df_routes):
    """Create 3D route heat map using PyDeck with elevation based on noise level"""
    if df_routes.empty:
        return None
    
    route_lines = []
    
    for idx, row in df_routes.iterrows():
        # Calculate elevation based on noise level (higher noise = higher elevation)
        # Scale: 50-100 dB -> 0-500m elevation
        elevation = (row['avg_noise_level'] - 50) * 10
        elevation = max(0, elevation)  # Ensure non-negative
        
        route_lines.append({
            'path': [
                [row['segment_start_lon'], row['segment_start_lat'], elevation],
                [row['segment_end_lon'], row['segment_end_lat'], elevation]
            ],
            'color': interpolate_route_color(row['avg_noise_level']),
            'width': 8,
            'street_name': row['street_name'],
            'noise': row['avg_noise_level'],
            'category': row['noise_category'],
            'neighborhood': row['neighborhood'],
            'elevation': elevation
        })
    
    df_lines = pd.DataFrame(route_lines)
    
    all_lats = df_routes[['segment_start_lat', 'segment_end_lat']].values.flatten()
    all_lons = df_routes[['segment_start_lon', 'segment_end_lon']].values.flatten()
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    layer = pdk.Layer(
        "PathLayer",
        df_lines,
        get_path='path',
        get_color='color',
        get_width='width',
        width_scale=1,
        width_min_pixels=3,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 100],
        extruded=True,
    )
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=13,
        pitch=45,  # Angle the camera for 3D effect
        bearing=0,  # Rotation
    )
    
    tooltip = {
        "html": "<b>{street_name}</b><br/>"
                "Avg Noise: <b>{noise:.1f} dB</b><br/>"
                "Category: <b>{category}</b><br/>"
                "Neighborhood: {neighborhood}<br/>"
                "Elevation: {elevation:.0f}m",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "14px"
        }
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
    )
    
    return deck

def create_recommended_routes_map(routes_data):
    """
    Create map showing recommended alternative routes with paths
    
    NEW: Added source and destination markers with icons
    """
    print("Routes data", routes_data, flush=True)
    print("Routes data", routes_data.get('routes'), flush=True)
    if not routes_data or not routes_data.get('routes'):
        return None
    
    try:
        routes = routes_data['routes']
        
        # ====================================================================
        # NEW: Extract origin and destination coordinates
        # ====================================================================
        origin_coords = None
        destination_coords = None
        
        # Get coordinates from the first route (all routes have same start/end)
        if routes and len(routes) > 0:
            first_route = routes[0]
            path_coords = first_route.get('path_coordinates', [])
            
            if path_coords and len(path_coords) >= 2:
                # First point is origin, last point is destination
                origin_coords = path_coords[0]
                destination_coords = path_coords[-1]
        
        # Build route lines directly from API response coordinates
        route_lines = []
        route_colors = [
            [46, 204, 113, 220],   # Green - Quietest
            [52, 152, 219, 220],   # Blue - Shortest
            [241, 196, 15, 220]    # Yellow - Balanced
        ]
        
        all_coords = []
        
        for route_idx, route in enumerate(routes):
            # Check if route has coordinates
            path_with_coords = route.get('path_coordinates', [])
            
            if not path_with_coords or len(path_with_coords) < 2:
                logger.warning(f"⚠️ Route {route_idx+1} has no coordinates or insufficient points")
                continue
            
            # Build path coordinates from API response
            path_coords = []
            for coord in path_with_coords:
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    lon, lat = float(coord[0]), float(coord[1])
                    path_coords.append([lon, lat])
                    all_coords.append([lon, lat])
            
            if len(path_coords) >= 2:
                route_lines.append({
                    'path': path_coords,
                    'color': route_colors[route_idx % len(route_colors)],
                    'width': 10 if route_idx == 0 else 6,  # Best route is thicker
                    'route_type': route['type'],
                    'noise': route['avg_noise_level'],
                    'distance': route['distance_meters'],
                    'time': route['estimated_time_minutes']
                })
                logger.info(f"   ✅ Route {route_idx+1} ({route['type']}): {len(path_coords)} points")
            else:
                logger.warning(f"   ⚠️ Route {route_idx+1} has insufficient valid coordinates: {len(path_coords)}")
        
        if not route_lines:
            logger.error("❌ No route lines could be created")
            return None
        
        if not all_coords:
            logger.error("❌ No coordinates in route lines")
            return None
        
        df_lines = pd.DataFrame(route_lines)
        
        # Calculate center from all coordinates
        center_lat = np.mean([c[1] for c in all_coords])
        center_lon = np.mean([c[0] for c in all_coords])
        
        logger.info(f"🎯 Map center: ({center_lat:.4f}, {center_lon:.4f})")
        logger.info(f"✅ Created {len(route_lines)} route lines with {len(all_coords)} total points")
        
        # ====================================================================
        # NEW: Create markers dataframe for origin and destination
        # ====================================================================
        markers_data = []
        
        if origin_coords:
            markers_data.append({
                'longitude': float(origin_coords[0]),
                'latitude': float(origin_coords[1]),
                'type': 'Origin',
                'icon': '🟢',  # Green circle for origin
                'color': [46, 204, 113, 255],  # Green
                'size': 300
            })
        
        if destination_coords:
            markers_data.append({
                'longitude': float(destination_coords[0]),
                'latitude': float(destination_coords[1]),
                'type': 'Destination',
                'icon': '🔴',  # Red circle for destination
                'color': [231, 76, 60, 255],  # Red
                'size': 300
            })
        
        # Create layers list
        layers = []
        
        # Add route path layer
        path_layer = pdk.Layer(
            "PathLayer",
            df_lines,
            get_path='path',
            get_color='color',
            get_width='width',
            width_scale=1,
            width_min_pixels=4,
            pickable=True,
            auto_highlight=True,
            highlight_color=[255, 255, 255, 150],
        )
        layers.append(path_layer)
        
        # ====================================================================
        # NEW: Add marker layer for origin and destination
        # ====================================================================
        if markers_data:
            df_markers = pd.DataFrame(markers_data)
            
            marker_layer = pdk.Layer(
                "ScatterplotLayer",
                df_markers,
                get_position='[longitude, latitude]',
                get_color='color',
                get_radius='size',
                pickable=True,
                opacity=1.0,
                stroked=True,
                filled=True,
                line_width_min_pixels=3,
                get_line_color=[255, 255, 255],  # White border
            )
            layers.append(marker_layer)
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=13,
            pitch=0,
        )
        
        # ====================================================================
        # NEW: Enhanced tooltip that shows marker type
        # ====================================================================
        tooltip = {}
        
        deck = pdk.Deck(
            layers=layers,  # Multiple layers now
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        )
        
        logger.info("✅ Route map created successfully with origin/destination markers")
        return deck
        
    except Exception as e:
        logger.error(f"❌ Error creating route map: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_noise_gauge_speedometer(current_noise, label="Live Noise"):
    """
    Speedometer-style gauge like in the image
    Shows current noise level with color zones
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_noise,
        title={'text': label, 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 70, 'increasing': {'color': "red"}},
        number={'suffix': " dB", 'font': {'size': 40, 'color': 'white'}},
        gauge={
            'axis': {
                'range': [None, 120],
                'tickwidth': 2,
                'tickcolor': "white",
                'tickfont': {'color': 'white', 'size': 14}
            },
            'bar': {'color': "rgba(255,255,255,0.8)", 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 3,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "#2ecc71", 'name': 'Quiet'},
                {'range': [50, 70], 'color': "#f1c40f", 'name': 'Moderate'},
                {'range': [70, 85], 'color': "#e67e22", 'name': 'Loud'},
                {'range': [85, 100], 'color': "#e74c3c", 'name': 'Very Loud'},
                {'range': [100, 120], 'color': "#c0392b", 'name': 'Extreme'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 6},
                'thickness': 0.85,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Arial'},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_hourly_pattern_chart(df):
    """
    Bar chart showing noise patterns throughout the day
    Similar to bottom chart in the image
    """
    if df.empty:
        return None
    
    # Extract hour from timestamp
    df_copy = df.copy()
    df_copy['hour'] = pd.to_datetime(df_copy['timestamp']).dt.hour
    
    # Aggregate by hour
    hourly = df_copy.groupby('hour')['noise_level'].agg(['mean', 'max', 'min']).reset_index()
    
    fig = go.Figure()
    
    # Add bars with color coding
    colors = ['#2ecc71' if x < 70 else '#f1c40f' if x < 85 else '#e74c3c' 
              for x in hourly['mean']]
    
    fig.add_trace(go.Bar(
        x=hourly['hour'],
        y=hourly['mean'],
        name='Average',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        text=[f"{x:.1f}" for x in hourly['mean']],
        textposition='outside',
        textfont=dict(color='white', size=10),
        hovertemplate='Hour: %{x}<br>Avg: %{y:.1f} dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='24-Hour Noise Pattern',
        xaxis_title='Hour of Day',
        yaxis_title='Noise Level (dB)',
        template='plotly_dark',
        height=300,
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            range=[0, 120]
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    return fig

def create_category_donut_chart(df):
    """
    Donut chart showing percentage of time in each noise category
    Like the pie charts in the image
    """
    if df.empty:
        return None
    
    # Categorize noise levels
    def categorize(level):
        if level < 55:
            return 'Quiet'
        elif level < 70:
            return 'Moderate'
        elif level < 85:
            return 'Loud'
        else:
            return 'Very Loud'
    
    df_copy = df.copy()
    df_copy['category'] = df_copy['noise_level'].apply(categorize)
    
    category_counts = df_copy['category'].value_counts()
    
    colors = {
        'Quiet': '#2ecc71',
        'Moderate': '#f1c40f',
        'Loud': '#e67e22',
        'Very Loud': '#e74c3c'
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.6,
        marker=dict(
            colors=[colors.get(cat, '#95a5a6') for cat in category_counts.index],
            line=dict(color='white', width=2)
        ),
        textposition='outside',
        textinfo='label+percent',
        textfont=dict(color='white', size=12),
        hovertemplate='%{label}<br>%{value} readings<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Noise Distribution',
        showlegend=False,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        annotations=[dict(
            text=f'{len(df)}<br>readings',
            x=0.5, y=0.5,
            font=dict(size=20, color='white'),
            showarrow=False
        )]
    )
    
    return fig

def create_circular_progress_metric(value, max_value, title, color="#3498db"):
    """
    Circular progress indicator like the red circles in the image
    """
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    
    fig = go.Figure()
    
    # Background circle
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(
            size=150,
            color='rgba(255,255,255,0.1)',
            line=dict(color='rgba(255,255,255,0.3)', width=3)
        ),
        hoverinfo='skip'
    ))
    
    # Value circle (filled based on percentage)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers+text',
        marker=dict(
            size=150 * (percentage / 100),
            color=color,
            opacity=0.8,
            line=dict(color=color, width=4)
        ),
        text=f"{value:.0f}",
        textposition='middle center',
        textfont=dict(size=30, color='white', family='Arial Black'),
        hovertemplate=f'{title}<br>{value:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[-1, 1]),
        yaxis=dict(visible=False, range=[-1, 1]),
        height=200,
        width=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=title, font=dict(size=12, color='white'), x=0.5, y=0.95)
    )
    
    return fig

def create_neighborhood_comparison_bars(df):
    """
    Horizontal bar chart comparing neighborhoods
    """
    if df.empty or 'neighborhood' not in df.columns:
        return None
    
    # Aggregate by neighborhood
    neighborhood_stats = df.groupby('neighborhood').agg({
        'noise_level': ['mean', 'max', 'count']
    }).round(1)
    
    neighborhood_stats.columns = ['avg', 'max', 'count']
    neighborhood_stats = neighborhood_stats.sort_values('avg', ascending=True).tail(10)
    
    fig = go.Figure()
    
    # Color based on noise level
    colors = ['#2ecc71' if x < 70 else '#f1c40f' if x < 85 else '#e74c3c' 
              for x in neighborhood_stats['avg']]
    
    fig.add_trace(go.Bar(
        y=neighborhood_stats.index,
        x=neighborhood_stats['avg'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=1)
        ),
        text=[f"{x:.1f} dB" for x in neighborhood_stats['avg']],
        textposition='outside',
        textfont=dict(color='white'),
        hovertemplate='%{y}<br>Avg: %{x:.1f} dB<br>Max: %{customdata[0]:.1f} dB<br>Readings: %{customdata[1]}<extra></extra>',
        customdata=neighborhood_stats[['max', 'count']].values
    ))
    
    fig.update_layout(
        title='Neighborhood Noise Levels',
        xaxis_title='Average Noise (dB)',
        template='plotly_dark',
        height=400,
        showlegend=False,
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    return fig

def create_time_series_area(df):
    """
    FIXED: Area chart showing actual noise trends over time
    
    Problem: Original was resampling incorrectly and had no data
    Solution: Properly handle timestamps and ensure data exists
    """
    if df.empty:
        st.warning("No data available for time series")
        return None
    
    # Make sure we have a proper dataframe
    df_copy = df.copy()
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    # Sort by time
    df_copy = df_copy.sort_values('timestamp')
    
    # Check if we have enough data
    if len(df_copy) < 2:
        st.warning("Need at least 2 data points for time series")
        return None
    
    # For real-time data, show last hour with 1-minute intervals
    time_range = (df_copy['timestamp'].max() - df_copy['timestamp'].min()).total_seconds()
    
    if time_range < 3600:  # Less than 1 hour
        # Show all points
        plot_df = df_copy[['timestamp', 'noise_level']].copy()
    else:
        # Resample to 5-minute intervals
        df_copy = df_copy.set_index('timestamp')
        plot_df = df_copy['noise_level'].resample('5T').mean().reset_index()
        plot_df = plot_df.dropna()  # Remove NaN values
    
    if len(plot_df) == 0:
        st.warning("No valid data after processing")
        return None
    
    fig = go.Figure()
    
    # Add filled area
    fig.add_trace(go.Scatter(
        x=plot_df['timestamp'],
        y=plot_df['noise_level'],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#3498db', width=3),
        fillcolor='rgba(52, 152, 219, 0.3)',
        name='Noise Level',
        hovertemplate='<b>%{x|%H:%M:%S}</b><br>Noise: %{y:.1f} dB<extra></extra>'
    ))
    
    # Add threshold lines with better visibility
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(241, 196, 15, 0.8)", line_width=2,
                  annotation_text="Moderate (70 dB)", annotation_position="right",
                  annotation=dict(font=dict(size=12, color="yellow")))
    fig.add_hline(y=85, line_dash="dash", line_color="rgba(230, 126, 34, 0.8)", line_width=2,
                  annotation_text="Loud (85 dB)", annotation_position="right",
                  annotation=dict(font=dict(size=12, color="orange")))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(231, 76, 60, 0.8)", line_width=2,
                  annotation_text="Very Loud (100 dB)", annotation_position="right",
                  annotation=dict(font=dict(size=12, color="red")))
    
    fig.update_layout(
        title=dict(
            text=f'Real-Time Noise Trends (Last {len(plot_df)} readings)',
            font=dict(size=18, color='white')
        ),
        xaxis_title='Time',
        yaxis_title='Noise Level (dB)',
        template='plotly_dark',
        height=400,
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            range=[40, 120],
            showgrid=True
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        showlegend=True,
        legend=dict(
            x=0.01, 
            y=0.99, 
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        ),
        hovermode='x unified'
    )
    
    return fig

def create_heatmap_calendar(df):
    """
    FIXED: Calendar heatmap showing actual noise patterns by day and hour
    
    Problem: Weird timestamps, solid colors, no variation
    Solution: Proper date/hour extraction and aggregation
    """
    if df.empty:
        st.warning("No data available for heatmap")
        return None
    
    df_copy = df.copy()
    
    # Convert timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    # Extract date and hour
    df_copy['date'] = df_copy['timestamp'].dt.date
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    
    # Check if we have multiple days
    unique_dates = df_copy['date'].nunique()
    if unique_dates < 2:
        st.info("📊 Need data from multiple days for weekly pattern heatmap")
        return None
    
    # Aggregate by date and hour
    heatmap_data = df_copy.groupby(['date', 'hour'])['noise_level'].mean().reset_index()
    
    # Pivot to create heatmap matrix
    pivot = heatmap_data.pivot(index='hour', columns='date', values='noise_level')
    
    # Get last 7 days
    if len(pivot.columns) > 7:
        pivot = pivot.iloc[:, -7:]
    
    # Check if we have data
    if pivot.empty or pivot.isna().all().all():
        st.warning("No valid data for heatmap after aggregation")
        return None
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[d.strftime('%b %d') for d in pivot.columns],
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0, '#2ecc71'],      # Green (quiet)
            [0.3, '#f1c40f'],    # Yellow (moderate)
            [0.6, '#e67e22'],    # Orange (loud)
            [1, '#e74c3c']       # Red (very loud)
        ],
        zmin=50,  # Set reasonable min
        zmax=100,  # Set reasonable max
        colorbar=dict(
            title=dict(text="dB", font=dict(color='white')),
            titleside="right",
            tickmode="linear",
            tick0=50,
            dtick=10,
            tickfont=dict(color='white')
        ),
        hovertemplate='<b>%{x}</b><br>Hour: %{y}<br>Noise: <b>%{z:.1f} dB</b><extra></extra>',
        showscale=True
    ))
    
    fig.update_layout(
        title=dict(
            text=f'Weekly Noise Patterns (Last {len(pivot.columns)} Days)',
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title='Date',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Hour of Day',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(128,128,128,0.2)'
        ),
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    return fig

def create_peak_times_indicator(df):
    """
    Indicator showing current peak noise times
    """
    if df.empty:
        return None
    
    df_copy = df.copy()
    df_copy['hour'] = pd.to_datetime(df_copy['timestamp']).dt.hour
    
    hourly_avg = df_copy.groupby('hour')['noise_level'].mean().sort_values(ascending=False)
    
    peak_hours = hourly_avg.head(3)
    
    html = "<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;'>"
    html += "<h3 style='margin: 0 0 15px 0; text-align: center;'>🔥 Peak Noise Times</h3>"
    
    for i, (hour, noise) in enumerate(peak_hours.items(), 1):
        time_str = f"{hour:02d}:00 - {(hour+1):02d}:00"
        color = "#e74c3c" if noise >= 85 else "#e67e22" if noise >= 70 else "#f1c40f"
        html += f"<div style='background: rgba(0,0,0,0.3); padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid {color};'>"
        html += f"<span style='font-size: 18px; font-weight: bold;'>#{i}</span> "
        html += f"<span style='font-size: 16px;'>{time_str}</span> "
        html += f"<span style='float: right; font-size: 18px; font-weight: bold;'>{noise:.1f} dB</span>"
        html += "</div>"
    
    html += "</div>"
    
    return html

def create_multi_series_comparison(df):
    """
    NEW: Multi-series chart comparing different neighborhoods/streets over time
    
    Shows top 5 locations with highest noise on same chart
    """
    if df.empty:
        return None
    
    df_copy = df.copy()
    
    # Convert timestamp
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    # Check if we have location data
    if 'neighborhood' not in df_copy.columns and 'street_name' not in df_copy.columns:
        st.info("No location data available for multi-series chart")
        return None
    
    # Use neighborhood if available, otherwise street
    location_col = 'neighborhood' if 'neighborhood' in df_copy.columns else 'street_name'
    
    # Find top 5 noisiest locations
    top_locations = df_copy.groupby(location_col)['noise_level'].mean().nlargest(5).index.tolist()
    
    if len(top_locations) == 0:
        return None
    
    # Filter for top locations
    df_filtered = df_copy[df_copy[location_col].isin(top_locations)].copy()
    
    # Sort by time
    df_filtered = df_filtered.sort_values('timestamp')
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Add trace for each location
    for i, location in enumerate(top_locations):
        location_data = df_filtered[df_filtered[location_col] == location].copy()
        
        # Resample to reduce data points
        if len(location_data) > 100:
            location_data = location_data.set_index('timestamp')
            location_data = location_data['noise_level'].resample('5T').mean().reset_index()
            location_data = location_data.dropna()
        
        if len(location_data) > 0:
            fig.add_trace(go.Scatter(
                x=location_data['timestamp'],
                y=location_data['noise_level'],
                mode='lines+markers',
                name=location,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{location}</b><br>Time: %{{x|%H:%M}}<br>Noise: %{{y:.1f}} dB<extra></extra>'
            ))
    
    # Add threshold reference
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,255,255,0.3)", 
                  annotation_text="70 dB", annotation_position="right")
    fig.add_hline(y=85, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="85 dB", annotation_position="right")
    
    fig.update_layout(
        title=dict(
            text='Multi-Location Noise Comparison (Top 5 Noisiest)',
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title='Time',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title='Noise Level (dB)',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            gridcolor='rgba(128,128,128,0.2)',
            range=[40, 120]
        ),
        template='plotly_dark',
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        ),
        hovermode='x unified'
    )
    
    return fig

def create_sensor_activity_grid(df):
    """
    NEW: Grid showing which sensors are active and their current status
    """
    if df.empty or 'sensor_id' not in df.columns:
        return None
    
    # Get latest reading from each sensor
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    latest_readings = df_copy.sort_values('timestamp').groupby('sensor_id').last().reset_index()
    
    # Calculate how old each reading is
    now = pd.Timestamp.now(tz=latest_readings['timestamp'].iloc[0].tz)
    latest_readings['age_seconds'] = (now - latest_readings['timestamp']).dt.total_seconds()
    
    # Categorize sensor status
    def get_status(age_seconds, noise_level):
        if age_seconds > 300:  # > 5 minutes
            return 'Offline', '#95a5a6'
        elif noise_level >= 90:
            return 'Critical', '#e74c3c'
        elif noise_level >= 80:
            return 'High', '#e67e22'
        elif noise_level >= 70:
            return 'Moderate', '#f1c40f'
        else:
            return 'Normal', '#2ecc71'
    
    latest_readings[['status', 'color']] = latest_readings.apply(
        lambda row: get_status(row['age_seconds'], row['noise_level']),
        axis=1,
        result_type='expand'
    )
    
    # Create grid layout (5 columns)
    n_cols = 5
    n_rows = (len(latest_readings) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"S{i+1}" for i in range(len(latest_readings))],
        specs=[[{'type': 'indicator'}] * n_cols for _ in range(n_rows)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    for idx, row in latest_readings.iterrows():
        row_idx = idx // n_cols + 1
        col_idx = idx % n_cols + 1
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=row['noise_level'],
                number={'suffix': " dB", 'font': {'size': 16, 'color': 'white'}},
                gauge={
                    'axis': {'range': [40, 120], 'visible': False},
                    'bar': {'color': row['color'], 'thickness': 1},
                    'bgcolor': 'rgba(0,0,0,0.3)',
                    'borderwidth': 0
                }
            ),
            row=row_idx,
            col=col_idx
        )
    
    fig.update_layout(
        title=dict(
            text=f'Sensor Network Status ({len(latest_readings)} sensors)',
            font=dict(size=18, color='white')
        ),
        template='plotly_dark',
        height=150 * n_rows,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig

def render_live_monitoring_tab(df_sensors, stats):
    """
    NEW TAB: Live Monitoring with creative visualizations
    Call this function in your main() when rendering tabs
    """
    st.markdown("### 🎛️ Live Noise Monitoring Dashboard")
    st.markdown("Real-time comprehensive noise analysis with advanced visualizations")
    
    if df_sensors.empty:
        st.warning("⏳ Waiting for sensor data...")
        return
    
    # ROW 1: Main Gauge + Current Stats
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        # Big speedometer gauge
        current_noise = df_sensors['noise_level'].iloc[-1] if not df_sensors.empty else 0
        gauge_fig = create_noise_gauge_speedometer(current_noise, "Current Noise Level")
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        # Circular progress - Daily Highest
        max_today = df_sensors['noise_level'].max()
        fig_max = create_circular_progress_metric(max_today, 120, "Daily Peak", "#e74c3c")
        st.plotly_chart(fig_max, use_container_width=True)
    
    with col3:
        # Circular progress - Weekly Highest
        avg_noise = df_sensors['noise_level'].mean()
        fig_avg = create_circular_progress_metric(avg_noise, 120, "Daily Avg", "#f1c40f")
        st.plotly_chart(fig_avg, use_container_width=True)
    
    with col4:
        # Circular progress - Monthly Highest
        reading_count = len(df_sensors)
        fig_count = create_circular_progress_metric(reading_count, 5000, "Readings", "#3498db")
        st.plotly_chart(fig_count, use_container_width=True)
    
    st.markdown("---")
    
    # ============================================================================
    # NEW: ROW 2 - Multi-Series Comparison Chart
    # ============================================================================
    st.markdown("#### 📈 Multi-Location Noise Comparison")
    multi_series_fig = create_multi_series_comparison(df_sensors)
    if multi_series_fig:
        st.plotly_chart(multi_series_fig, use_container_width=True)
    else:
        st.info("📊 Multi-series chart requires data from multiple locations")
    
    st.markdown("---")
    
    # ROW 3: Time Series + Category Distribution (KEEP AS IS, but use FIXED version)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Real-time trend line - USE FIXED VERSION
        trend_fig = create_time_series_area(df_sensors)  # ← CHANGED
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.warning("Unable to generate time series")
    
    with col2:
        # Donut chart
        donut_fig = create_category_donut_chart(df_sensors)
        if donut_fig:
            st.plotly_chart(donut_fig, use_container_width=True)
    
    st.markdown("---")
    
    # ROW 4: Hourly Pattern + Neighborhood Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # 24-hour pattern bars
        hourly_fig = create_hourly_pattern_chart(df_sensors)
        if hourly_fig:
            st.plotly_chart(hourly_fig, use_container_width=True)
    
    with col2:
        # Neighborhood bars
        neighborhood_fig = create_neighborhood_comparison_bars(df_sensors)
        if neighborhood_fig:
            st.plotly_chart(neighborhood_fig, use_container_width=True)
    
    st.markdown("---")
    
    # ROW 5: Heatmap Calendar - USE FIXED VERSION
    st.markdown("#### 🗓️ Weekly Noise Patterns")
    heatmap_fig = create_heatmap_calendar(df_sensors)  # ← CHANGED
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("📅 Heatmap requires data from multiple days. Keep system running to build history!")
    
    st.markdown("---")
    
    # ============================================================================
    # NEW: ROW 6 - Sensor Activity Grid
    # ============================================================================
    st.markdown("#### 🔌 Sensor Network Status")
    sensor_grid = create_sensor_activity_grid(df_sensors)
    if sensor_grid:
        st.plotly_chart(sensor_grid, use_container_width=True)
    else:
        st.info("📡 Sensor grid requires sensor_id data")
    
    st.markdown("---")
    
    # ROW 7: Peak Times Indicator
    peak_html = create_peak_times_indicator(df_sensors)
    if peak_html:
        st.markdown(peak_html, unsafe_allow_html=True)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🇺🇸 Milwaukee Traffic Noise Monitoring System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete AI-Powered Noise Analysis & Route Optimization Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Milwaukee Flag (People's Flag) or US Flag - using US Flag for reliability
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a4/Flag_of_the_United_States.svg/320px-Flag_of_the_United_States.svg.png", width=100)
        st.title("🎛️ Controls")
        
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        time_window = st.selectbox(
            "Historical Data Window",
            [("Last Hour", 1), ("Last 6 Hours", 6), ("Last 24 Hours", 24)],
            format_func=lambda x: x[0]
        )
        
        st.markdown("---")
        st.subheader("ℹ️ System Status")
        
        # Check API
        try:
            response = requests.get("http://alternative-routes-api:5000/health", timeout=2)
            api_status = "✅ Online" if response.status_code == 200 else "❌ Offline"
        except:
            api_status = "❌ Offline"
        
        st.info(f"""
        **Services:**
        - 📊 Database: ✅ Connected
        - 🗺️ Route API: {api_status}
        - 🔮 Predictions: ✅ Active
        - 😰 Stress Zones: ✅ Active
        - 📈 Correlations: ✅ Active
        
        **Noise Levels:**
        - 🟢 <70 dB: Low
        - 🟡 70-80 dB: Moderate
        - 🟠 80-90 dB: High
        - 🔴 ≥90 dB: Critical
        """)
        
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Fetch all data
    with st.spinner("Loading comprehensive data..."):
        stats = fetch_statistics()
        df_sensors = fetch_latest_readings(hours=time_window[1])
        df_routes = fetch_route_segments(hours=2)
        df_predictions = fetch_noise_predictions(hours_ahead=24)
        df_stress = fetch_stress_zones()
        df_correlations = fetch_correlations()
        df_recommendations = fetch_route_recommendations()
    
    # Metrics Row
    if stats and len(stats) > 0:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            total_readings = stats.get('total_readings', 0)
            st.metric(
                "Total Readings",
                f"{int(total_readings):,}" if total_readings else "0"
            )
        
        with col2:
            active_sensors = stats.get('active_sensors', 0)
            st.metric(
                "Active Sensors",
                int(active_sensors) if active_sensors else 0
            )
        
        with col3:
            avg_noise = stats.get('avg_noise', 0)
            if avg_noise and avg_noise > 0:
                st.metric(
                    "Avg Noise",
                    f"{float(avg_noise):.1f} dB",
                    delta=f"{float(avg_noise) - 75:.1f}"
                )
            else:
                st.metric(
                    "Avg Noise",
                    "N/A"
                )
        
        with col4:
            st.metric(
                "Predictions",
                len(df_predictions) if df_predictions is not None else 0
            )
        
        with col5:
            stress_count = len(df_stress) if df_stress is not None else 0
            st.metric(
                "Stress Zones",
                stress_count,
                delta="High" if stress_count > 10 else None
            )
        
        with col6:
            critical = stats.get('critical_count', 0)
            st.metric(
                "Critical Alerts",
                int(critical) if critical else 0
            )
    else:
        st.info("⏳ Waiting for data... System is initializing. Please wait 1-2 minutes.")
    
    st.markdown("---")
    
    # Main Content - Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
       "🗺️ Route Heat Map",
       "🔮 Noise Predictions", 
       "😰 Stress Zones",
       "📊 Correlation Analysis",
       "🛣️ Route Recommendations",
       "📈 Advanced Analytics",
       "🎛️ Live Monitoring" 
    ])
    
    # ========================================================================
    # TAB 1: ROUTE HEAT MAP
    # ========================================================================
    with tab1:
        st.subheader("🛣️ Real-Time Route Noise Heat Map")
        
        if df_routes.empty:
            st.warning("⚠️ No route data available. Wait for Spark Consumer to generate segments.")
        else:
            # Add 3D view toggle
            col_toggle1, col_toggle2 = st.columns([1, 4])
            with col_toggle1:
                view_3d = st.checkbox("🏔️ 3D View", value=False)
            with col_toggle2:
                if view_3d:
                    st.caption("3D view shows noise levels as elevated paths - higher = louder")
            
            col_map, col_stats = st.columns([3, 1])
            
            with col_map:
                if view_3d:
                    # Create 3D version
                    deck_3d = create_route_heatmap_3d(df_routes)
                    if deck_3d:
                        st.pydeck_chart(deck_3d, use_container_width=True)
                else:
                    # Original 2D version
                    deck = create_route_heatmap(df_routes)
                    if deck:
                        st.pydeck_chart(deck, use_container_width=True)
            
            with col_stats:
                st.markdown("### 📊 Route Statistics")
                
                category_counts = df_routes['noise_category'].value_counts()
                
                for category in ['critical', 'high', 'moderate', 'low']:
                    count = category_counts.get(category, 0)
                    color = get_noise_color_hex(
                        {'critical': 95, 'high': 85, 'moderate': 75, 'low': 65}[category]
                    )
                    st.markdown(
                        f'<div style="background-color:{color};padding:10px;border-radius:5px;margin:5px 0;color:white;">'
                        f'<b>{category.upper()}</b><br/>{count} segments</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown("---")
                st.metric("Total Segments", len(df_routes))
                st.metric("Avg Noise", f"{df_routes['avg_noise_level'].mean():.1f} dB")
            
            with st.expander("📋 Route Details Table"):
                # Add filters to reduce data
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                
                with col_filter1:
                    selected_category = st.multiselect(
                        "Noise Category",
                        options=['critical', 'high', 'moderate', 'low'],
                        default=['critical', 'high', 'moderate', 'low'],
                        key='route_category_filter'
                    )
                
                with col_filter2:
                    neighborhoods = ['All'] + sorted(df_routes['neighborhood'].unique().tolist())
                    selected_neighborhood = st.selectbox("Neighborhood", neighborhoods, key='route_neighborhood_filter')
                
                with col_filter3:
                    show_limit = st.slider("Show top rows", 10, 500, 100, step=10, key='route_limit')
                
                # Apply filters
                display_df = df_routes[[
                    'street_name', 'neighborhood', 'avg_noise_level',
                    'noise_category', 'segment_length_meters'
                ]].copy()
                
                # Filter by category
                display_df = display_df[display_df['noise_category'].isin(selected_category)]
                
                # Filter by neighborhood
                if selected_neighborhood != 'All':
                    display_df = display_df[display_df['neighborhood'] == selected_neighborhood]
                
                # Rename columns
                display_df.columns = ['Street', 'Neighborhood', 'Avg Noise (dB)', 'Category', 'Length (m)']
                
                # Sort and limit
                display_df = display_df.sort_values('Avg Noise (dB)', ascending=False).head(show_limit)
                
                st.caption(f"Showing {len(display_df)} of {len(df_routes)} total segments")
                
                # Display WITHOUT styling (fast, no errors)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Optional: Add download button for full data
                csv = df_routes.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name="route_segments.csv",
                    mime="text/csv",
                    key='download_routes'
                )
    
    # ========================================================================
    # TAB 2: NOISE PREDICTIONS
    # ========================================================================
    with tab2:
        st.subheader("🔮 AI-Powered Noise Predictions")
        
        if df_predictions.empty:
            st.warning("⚠️ No predictions available yet. Prediction service runs every 15 minutes.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Predictions",
                    len(df_predictions)
                )
            
            with col2:
                avg_predicted = df_predictions['predicted_noise_level'].mean()
                st.metric(
                    "Avg Predicted Noise",
                    f"{avg_predicted:.1f} dB"
                )
            
            with col3:
                max_predicted = df_predictions['predicted_noise_level'].max()
                st.metric(
                    "Peak Predicted",
                    f"{max_predicted:.1f} dB",
                    delta="High" if max_predicted > 90 else None
                )
            
            with col4:
                avg_confidence = df_predictions['confidence_score'].mean()
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.0%}"
                )
            
            st.markdown("---")
            
            # Timeline
            fig_timeline = create_prediction_timeline(df_predictions)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Two columns
            col1, col2 = st.columns(2)
            
            with col1:
                fig_heatmap = create_prediction_heatmap(df_predictions)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col2:
                deck_pred = create_prediction_map(df_predictions)
                if deck_pred:
                    st.pydeck_chart(deck_pred, use_container_width=True)
            
            # Detailed table
            with st.expander("📋 Prediction Details"):
                pred_display = df_predictions[[
                    'street_name', 'forecast_timestamp', 'forecast_horizon',
                    'predicted_noise_level', 'prediction_interval_lower',
                    'prediction_interval_upper', 'confidence_score'
                ]].copy()
                
                pred_display.columns = [
                    'Street', 'Forecast Time', 'Horizon (h)',
                    'Predicted (dB)', 'Lower CI', 'Upper CI', 'Confidence'
                ]
                
                st.dataframe(pred_display, use_container_width=True, height=400)
    
    # ========================================================================
    # TAB 3: STRESS ZONES
    # ========================================================================
    with tab3:
        st.subheader("😰 Predicted Stress Zones")
        
        if df_stress.empty:
            st.warning("⚠️ No stress zones detected. This is good news!")
        else:
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Stress Zones",
                    len(df_stress)
                )
            
            with col2:
                critical_zones = len(df_stress[df_stress['alert_level'] == 'critical'])
                st.metric(
                    "Critical Zones",
                    critical_zones,
                    delta="⚠️" if critical_zones > 0 else None
                )
            
            with col3:
                avg_stress = df_stress['predicted_stress_level'].mean()
                st.metric(
                    "Avg Stress Level",
                    f"{avg_stress:.1f}"
                )
            
            with col4:
                max_stress = df_stress['predicted_stress_level'].max()
                st.metric(
                    "Peak Stress",
                    f"{max_stress:.1f}"
                )
            
            st.markdown("---")
            
            # Map and gauge
            col1, col2 = st.columns([2, 1])
            
            with col1:
                deck_stress = create_stress_zone_map(df_stress)
                if deck_stress:
                    st.pydeck_chart(deck_stress, use_container_width=True)
            
            with col2:
                if not df_stress.empty:
                    worst_zone = df_stress.loc[df_stress['predicted_stress_level'].idxmax()]
                    fig_gauge = create_stress_gauge(
                        worst_zone['predicted_stress_level'],
                        worst_zone['alert_level']
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    st.markdown(f"**Worst Zone:** {worst_zone['zone_name']}")
                    st.markdown(f"**Action:** {worst_zone['recommended_action']}")
            
            # Breakdown
            fig_breakdown = create_stress_breakdown(df_stress)
            if fig_breakdown:
                st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Alert zones table
            with st.expander("🚨 Alert Zones Details"):
                alert_zones = df_stress[df_stress['alert_level'].isin(['high', 'critical'])].copy()
                
                if not alert_zones.empty:
                    # Limit to top 100 by stress level
                    alert_zones = alert_zones.nlargest(100, 'predicted_stress_level')
                    
                    alert_display = alert_zones[[
                        'zone_name', 'predicted_stress_level', 'alert_level',
                        'predicted_noise_contribution', 'predicted_sentiment_contribution',
                        'recommended_action'
                    ]].copy()
                    
                    alert_display.columns = [
                        'Zone', 'Stress Level', 'Alert',
                        'Noise Contrib.', 'Sentiment Contrib.', 'Recommended Action'
                    ]
                    
                    st.caption(f"Showing top {len(alert_display)} zones by stress level")
                    
                    # Display WITHOUT styling
                    st.dataframe(
                        alert_display,
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.success("✅ No high or critical stress zones detected!")
    
    # ========================================================================
    # TAB 4: CORRELATION ANALYSIS
    # ========================================================================
    with tab4:
        st.subheader("📊 Noise-Sentiment Correlation Analysis")
        
        if df_correlations.empty:
            st.warning("⚠️ No correlation data available. Analysis runs every 5 minutes.")
        else:
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Analyses",
                    len(df_correlations)
                )
            
            with col2:
                avg_corr = df_correlations['correlation_coefficient'].mean()
                corr_type = "Negative" if avg_corr < -0.3 else "Positive" if avg_corr > 0.3 else "Weak"
                st.metric(
                    "Avg Correlation",
                    f"{avg_corr:.3f}",
                    delta=corr_type
                )
            
            with col3:
                strong_corr = len(df_correlations[abs(df_correlations['correlation_coefficient']) > 0.5])
                st.metric(
                    "Strong Correlations",
                    strong_corr
                )
            
            with col4:
                significant = len(df_correlations[df_correlations['statistical_significance'] < 0.05])
                st.metric(
                    "Statistically Significant",
                    significant
                )
            
            st.markdown("---")
            
            # Main scatter plot
            fig_scatter = create_correlation_scatter(df_correlations)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Two columns
            col1, col2 = st.columns(2)
            
            with col1:
                fig_heatmap_corr = create_correlation_heatmap(df_correlations)
                if fig_heatmap_corr:
                    st.plotly_chart(fig_heatmap_corr, use_container_width=True)
            
            with col2:
                fig_dist = create_correlation_distribution(df_correlations)
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            # Insights
            st.markdown("### 🔍 Key Insights")
            
            # Strongest positive correlation
            if not df_correlations.empty:
                strongest_positive = df_correlations.loc[df_correlations['correlation_coefficient'].idxmax()]
                strongest_negative = df_correlations.loc[df_correlations['correlation_coefficient'].idxmin()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(
                        f'<div class="correlation-positive">'
                        f'<b>Strongest Positive Correlation</b><br/>'
                        f'{strongest_positive["location_name"]}<br/>'
                        f'r = {strongest_positive["correlation_coefficient"]:.3f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        f'<div class="correlation-negative">'
                        f'<b>Strongest Negative Correlation</b><br/>'
                        f'{strongest_negative["location_name"]}<br/>'
                        f'r = {strongest_negative["correlation_coefficient"]:.3f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            # Detailed table
            with st.expander("📋 Correlation Details"):
                # Limit to top 100 correlations
                corr_limited = df_correlations.head(100)
                
                corr_display = corr_limited[[
                    'location_name', 'time_window', 'avg_noise_level',
                    'avg_sentiment_score', 'correlation_coefficient',
                    'sample_size', 'statistical_significance'
                ]].copy()
                
                corr_display.columns = [
                    'Location', 'Time Window', 'Avg Noise', 'Avg Sentiment',
                    'Correlation', 'Sample Size', 'p-value'
                ]
                
                st.caption(f"Showing {len(corr_display)} of {len(df_correlations)} total correlations")
                
                # Display WITHOUT styling
                st.dataframe(
                    corr_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download option
                csv = df_correlations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Correlations",
                    data=csv,
                    file_name="correlations.csv",
                    mime="text/csv",
                    key='download_correlations'
                )
    
    # ========================================================================
    # TAB 5: ROUTE RECOMMENDATIONS
    # ========================================================================
    with tab5:
        render_route_recommendations_tab(df_recommendations)
    
    # ========================================================================
    # TAB 6: ADVANCED ANALYTICS
    # ========================================================================
    with tab6:
        st.subheader("📈 Advanced Analytics Dashboard")
        
        # Multi-metric comparison
        st.markdown("### 📊 Multi-Metric Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction accuracy over time
            if not df_predictions.empty:
                pred_by_horizon = df_predictions.groupby('forecast_horizon').agg({
                    'predicted_noise_level': 'mean',
                    'confidence_score': 'mean'
                }).reset_index()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=pred_by_horizon['forecast_horizon'],
                    y=pred_by_horizon['predicted_noise_level'],
                    name='Predicted Noise',
                    mode='lines+markers',
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_by_horizon['forecast_horizon'],
                    y=pred_by_horizon['confidence_score'] * 100,
                    name='Confidence',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title='Prediction Performance by Forecast Horizon',
                    xaxis_title='Hours Ahead',
                    yaxis_title='Predicted Noise (dB)',
                    yaxis2=dict(
                        title='Confidence (%)',
                        overlaying='y',
                        side='right'
                    ),
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stress vs Noise scatter
            if not df_stress.empty:
                # Create a positive size column (absolute value + small constant to avoid zero)
                df_stress_plot = df_stress.copy()
                df_stress_plot['sentiment_magnitude'] = abs(df_stress_plot['predicted_sentiment_contribution']) + 1
                
                fig = px.scatter(
                    df_stress_plot,
                    x='predicted_noise_contribution',
                    y='predicted_stress_level',
                    color='alert_level',
                    size='sentiment_magnitude',
                    hover_data=['zone_name', 'predicted_sentiment_contribution'],
                    title='Stress Level vs Noise Contribution',
                    labels={
                        'predicted_noise_contribution': 'Noise Contribution',
                        'predicted_stress_level': 'Stress Level',
                        'sentiment_magnitude': 'Sentiment Impact'
                    },
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # System performance metrics
        st.markdown("### ⚡ System Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            data_coverage = (len(df_sensors) / (stats.get('active_sensors', 1) * 12)) * 100  # Assuming 12 readings/hour
            st.metric(
                "Data Coverage",
                f"{min(data_coverage, 100):.1f}%"
            )
        
        with col2:
            prediction_rate = len(df_predictions) / max(len(df_sensors.groupby(['street_name', 'neighborhood'])), 1)
            st.metric(
                "Prediction Rate",
                f"{prediction_rate:.1f}x"
            )
        
        with col3:
            if not df_correlations.empty:
                correlation_quality = (df_correlations['statistical_significance'] < 0.05).mean() * 100
                st.metric(
                    "Correlation Quality",
                    f"{correlation_quality:.0f}%"
                )
        
        with col4:
            if not df_routes.empty:
                route_density = len(df_routes) / len(df_routes.groupby('street_name'))
                st.metric(
                    "Route Density",
                    f"{route_density:.1f}"
                )
        
        # Raw data access
        with st.expander("🔬 Raw Data Explorer"):
            data_choice = st.selectbox(
                "Select Dataset",
                ["Predictions", "Stress Zones", "Correlations", "Routes", "Recommendations"]
            )
            
            if data_choice == "Predictions" and not df_predictions.empty:
                # Limit display
                show_count = st.slider("Rows to display", 10, 1000, 100, step=10, key='raw_data_limit')
                st.caption(f"Showing {min(show_count, len(df_predictions))} of {len(df_predictions)} rows")
                st.dataframe(df_predictions.head(show_count), use_container_width=True, height=400)
                
                # Download full data
                csv = df_predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"{data_choice.lower()}.csv",
                    mime="text/csv",
                    key=f'download_{data_choice}'
                )
                
            elif data_choice == "Stress Zones" and not df_stress.empty:
                show_count = st.slider("Rows to display", 10, 1000, 100, step=10, key='raw_data_limit')
                st.caption(f"Showing {min(show_count, len(df_stress))} of {len(df_stress)} rows")
                st.dataframe(df_stress.head(show_count), use_container_width=True, height=400)
                
                csv = df_stress.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"{data_choice.lower()}.csv",
                    mime="text/csv",
                    key=f'download_{data_choice}'
                )
                
            elif data_choice == "Correlations" and not df_correlations.empty:
                show_count = st.slider("Rows to display", 10, 1000, 100, step=10, key='raw_data_limit')
                st.caption(f"Showing {min(show_count, len(df_correlations))} of {len(df_correlations)} rows")
                st.dataframe(df_correlations.head(show_count), use_container_width=True, height=400)
                
                csv = df_correlations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"{data_choice.lower()}.csv",
                    mime="text/csv",
                    key=f'download_{data_choice}'
                )
                
            elif data_choice == "Routes" and not df_routes.empty:
                show_count = st.slider("Rows to display", 10, 1000, 100, step=10, key='raw_data_limit')
                st.caption(f"Showing {min(show_count, len(df_routes))} of {len(df_routes)} rows")
                st.dataframe(df_routes.head(show_count), use_container_width=True, height=400)
                
                csv = df_routes.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"{data_choice.lower()}.csv",
                    mime="text/csv",
                    key=f'download_{data_choice}'
                )
                
            elif data_choice == "Recommendations" and not df_recommendations.empty:
                show_count = st.slider("Rows to display", 10, 1000, 100, step=10, key='raw_data_limit')
                st.caption(f"Showing {min(show_count, len(df_recommendations))} of {len(df_recommendations)} rows")
                st.dataframe(df_recommendations.head(show_count), use_container_width=True, height=400)
                
                csv = df_recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=csv,
                    file_name=f"{data_choice.lower()}.csv",
                    mime="text/csv",
                    key=f'download_{data_choice}'
                )
            else:
                st.info("No data available for this dataset")
        with tab7:
            render_live_monitoring_tab(df_sensors, stats)
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        st.caption("🇺🇸 Milwaukee, Wisconsin")
    
    with footer_col2:
        db_status = '✅ Connected' if get_db_connection() else '❌ Disconnected'
        st.caption(f"📡 Database: {db_status}")
    
    with footer_col3:
        st.caption(f"🗺️ Route API: {api_status}")
    
    with footer_col4:
        st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    logger.info("🚀 Starting Complete Milwaukee Noise Monitoring Dashboard")
    main()