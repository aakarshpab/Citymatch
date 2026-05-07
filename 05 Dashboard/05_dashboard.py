import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CityMatch AI | Delhi", layout="wide", page_icon="🏙️")

# --- CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #2e7d32; }
    .stDataFrame { border-radius: 12px; }
    .route-card { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #2ca02c; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clustered_results_v2.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("❌ 'clustered_results.csv' not found!")
    st.stop()

# --- 2. SIDEBAR: USER INPUTS ---
with st.sidebar:
    st.title("🎯 Preference Tuner")
    w_air = st.slider("🍃 Clean Air", 0, 10, 8)
    w_traffic = st.slider("🚗 Low Traffic", 0, 10, 6)
    w_rent = st.slider("💰 Affordable Rent", 0, 10, 7)
    w_school = st.slider("🎓 Schools", 0, 10, 5)
    w_hosp = st.slider("🏥 Hospitals", 0, 10, 5)
    w_metro = st.slider("🚇 Metro Access", 0, 10, 6)
    weights = [w_air, w_traffic, w_rent, w_school, w_hosp, w_metro]

# --- 3. RECOMMENDER ENGINE ---
def calculate_recommendations(df, weights):
    features = ['PM25_Winter', 'Traffic_Index', 'Rent_Price', 'num_schools', 'num_hospitals', 'num_metro_stations']
    scaler = MinMaxScaler()
    norm_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    
    # Invert cost/negative features
    norm_df['PM25_Winter'] = 1 - norm_df['PM25_Winter']
    norm_df['Traffic_Index'] = 1 - norm_df['Traffic_Index']
    norm_df['Rent_Price'] = 1 - norm_df['Rent_Price']

    total_weight = sum(weights) if sum(weights) > 0 else 1
    user_vector = np.array([w/total_weight for w in weights]).reshape(1, -1)
    scores = cosine_similarity(user_vector, norm_df)[0]
    df['Match_Score'] = scores * 100 
    return df.sort_values(by='Match_Score', ascending=False).reset_index(drop=True)

ranked_df = calculate_recommendations(df.copy(), weights)
top_match = ranked_df.iloc[0]

# --- 4. MAIN DASHBOARD UI ---
st.title("🏙️ CityMatch: Delhi AI Recommender")

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("🏆 Best Match", top_match['neighborhood'])
m2.metric("✅ Match Score", f"{top_match['Match_Score']:.1f}%")
m3.metric("💵 Est. Rent", f"₹{int(top_match['Rent_Price'])}")
m4.metric("🤖 AI Cluster", f"Group {int(top_match['Cluster_ID'])}")

st.divider()

# Layout: Map and Radar
col_left, col_right = st.columns([1.2, 0.8])
with col_left:
    st.subheader("📍 Recommended Neighborhoods")
    m = folium.Map(location=[top_match['latitude'], top_match['longitude']], zoom_start=12)
    for idx, row in ranked_df.head(5).iterrows():
        folium.Marker([row['latitude'], row['longitude']], popup=row['neighborhood'], icon=folium.Icon(color='green' if idx==0 else 'blue')).add_to(m)
    st_folium(m, height=400, width=None, use_container_width=True)

with col_right:
    st.subheader("📊 Your Priority Profile")
    categories = ['Air', 'Traffic', 'Rent', 'Schools', 'Hospitals', 'Metro']
    fig = go.Figure(data=go.Scatterpolar(r=weights, theta=categories, fill='toself', line_color='#2ca02c'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

# --- 5. NEW SECTION: SHORTEST ROUTES TO AMENITIES ---
st.divider()
st.header("🚶 Proximity to Essential Services")
st.write(f"Showing distances from the center of **{top_match['neighborhood']}**.")

# NOTE: In a real project, you would have 'school_lat', 'school_long' etc. in your CSV.
# Here we calculate relative proximity based on your neighborhood center.
amenities = [
    {"type": "School", "name": "National Victor Public School", "coords": [28.6829, 77.2801]},
    {"type": "Hospital", "name": "CM Patel Hospital", "coords": [28.6891, 77.2885]},
    {"type": "Metro", "name": "Jaffrabad Metro Station", "coords": [28.6827, 77.2747]}
]

route_cols = st.columns(3)
for i, item in enumerate(amenities):
    with route_cols[i]:
        st.markdown(f"**{item['type']}**: {item['name']}")
        
        # Create a mini map for the route
        mini_m = folium.Map(location=top_match[['latitude', 'longitude']], zoom_start=14, zoom_control=False)
        
        # Add Home and Amenity Markers
        folium.Marker([top_match['latitude'], top_match['longitude']], icon=folium.Icon(color='red', icon='home')).add_to(mini_m)
        folium.Marker(item['coords'], icon=folium.Icon(color='blue', icon='info-sign')).add_to(mini_m)
        
        # Draw shortest path (straight line)
        folium.PolyLine([[top_match['latitude'], top_match['longitude']], item['coords']], color="green", weight=3, opacity=0.8).add_to(mini_m)
        
        st_folium(mini_m, height=200, key=f"route_map_{i}", use_container_width=True)
        
        # Google Maps link
        url = f"https://www.google.com/maps/dir/{top_match['latitude']},{top_match['longitude']}/{item['coords'][0]},{item['coords'][1]}"
        st.link_button(f"Open Route in Maps", url)

# --- 6. DATA TABLE ---
st.divider()
st.subheader("📋 Top 10 Candidates Comparison")
top_10_df = ranked_df[['neighborhood', 'Match_Score', 'Rent_Price', 'PM25_Winter', 'nearest_station']].head(10).copy()
top_10_df.index = top_10_df.index + 1
st.dataframe(top_10_df.style.background_gradient(subset=['Match_Score'], cmap='Greens'), use_container_width=True)