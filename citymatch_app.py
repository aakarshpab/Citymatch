import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CityMatch | Delhi Neighbourhood AI",
    layout="wide",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0d0f14; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #22c55e !important;
}

/* ── Headers ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #f9fafb !important; font-family: 'Syne', sans-serif !important; font-size: 1.5rem !important; }

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #166534, #15803d);
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    padding: 0.6rem 1.4rem;
    border-radius: 12px;
    border: 1px solid #22c55e33;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b7280;
    margin-bottom: 0.5rem;
}

/* ── Factor bar ── */
.factor-row {
    display: flex;
    align-items: center;
    margin: 6px 0;
    gap: 10px;
}
.factor-label { color: #9ca3af; font-size: 0.82rem; width: 110px; flex-shrink: 0; }
.factor-bar-bg { flex: 1; background: #1f2937; border-radius: 4px; height: 7px; }
.factor-bar-fill { height: 7px; border-radius: 4px; background: linear-gradient(90deg, #166534, #22c55e); }
.factor-value { color: #e2e8f0; font-size: 0.82rem; width: 36px; text-align: right; flex-shrink: 0; }

/* ── Neighbourhood card ── */
.nbhd-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nbhd-name { font-family: 'Syne', sans-serif; font-weight: 600; color: #f3f4f6; font-size: 0.95rem; }
.nbhd-meta { color: #6b7280; font-size: 0.78rem; margin-top: 2px; }
.nbhd-score { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem; }

/* ── Divider ── */
hr { border-color: #1f2937 !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("clustered_results_v4.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("clustered_results_v3.csv")
        except FileNotFoundError:
            return None

    # Ensure scaled columns exist — regenerate if missing
    features_to_scale = ['PM25_Winter', 'Traffic_Index', 'Rent_Price',
                         'num_schools', 'num_hospitals', 'num_metro_stations']
    missing = [f for f in features_to_scale if f"{f}_scaled" not in df.columns]
    if missing:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features_to_scale])
        for i, col in enumerate(features_to_scale):
            df[f"{col}_scaled"] = scaled[:, i]

    return df

df = load_data()

if df is None:
    st.error("❌ Dataset not found. Please make sure clustered_results_v3.csv is in the project folder.")
    st.stop()
df = df[df['neighborhood'] != 'Najafgarh'].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════
# SCORING ENGINE  (fixed: weighted sum, not cosine similarity)
# ══════════════════════════════════════════════════════════════
# IMPORTANT: Lower is better for PM25, Traffic, Rent → invert their scaled values.
# Higher is better for Schools, Hospitals, Metro → use scaled value directly.
#
# Formula per neighbourhood:
#   score = Σ (weight_i × adjusted_feature_i) / Σ weight_i  × 100
#
def calculate_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    df = df.copy()

    # Adjusted feature values (all now "higher = better")
    df['_air']    = 1 - df['PM25_Winter_scaled']       # lower PM2.5 → better air
    df['_traffic']= 1 - df['Traffic_Index_scaled']     # lower congestion → better
    df['_rent']   = 1 - df['Rent_Price_scaled']        # lower rent → more affordable
    df['_school'] = df['num_schools_scaled']
    df['_hosp']   = df['num_hospitals_scaled']
    df['_metro']  = df['num_metro_stations_scaled']

    total_weight = sum(weights.values())
    if total_weight == 0:
        df['Match_Score'] = 0.0
        return df.sort_values('Match_Score', ascending=False).reset_index(drop=True)

    df['Match_Score'] = (
        weights['air']     * df['_air']     +
        weights['traffic'] * df['_traffic'] +
        weights['rent']    * df['_rent']    +
        weights['school']  * df['_school']  +
        weights['hosp']    * df['_hosp']    +
        weights['metro']   * df['_metro']
    ) / total_weight * 100

    return df.sort_values('Match_Score', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏙️ CityMatch")
    st.markdown("<p style='color:#6b7280;font-size:0.82rem;margin-top:-8px;'>Delhi Neighbourhood Recommender</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 Your Priorities")
    st.caption("Drag sliders to weight what matters most to you.")

    w_air     = st.slider("🍃 Clean Air",       0, 10, 8)
    w_traffic = st.slider("🚗 Low Traffic",      0, 10, 6)
    w_rent    = st.slider("💰 Affordable Rent",  0, 10, 7)
    w_school  = st.slider("🎓 Schools Nearby",   0, 10, 5)
    w_hosp    = st.slider("🏥 Hospitals Nearby", 0, 10, 5)
    w_metro   = st.slider("🚇 Metro Access",     0, 10, 6)

    st.markdown("---")
    st.caption("Scores are computed as a weighted sum across all 6 liveability factors. Lower rent, traffic & pollution are rewarded.")

weights = {
    'air': w_air, 'traffic': w_traffic, 'rent': w_rent,
    'school': w_school, 'hosp': w_hosp, 'metro': w_metro
}

ranked_df  = calculate_scores(df, weights)
top        = ranked_df.iloc[0]
top10      = ranked_df.head(10)


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-family:Syne,sans-serif;font-size:2rem;color:#f9fafb;margin-bottom:0;'>
    🏙️ CityMatch <span style='color:#22c55e;'>Delhi</span>
</h1>
<p style='color:#6b7280;margin-top:2px;font-size:0.9rem;'>
    AI-powered neighbourhood matching based on your liveability priorities
</p>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ══════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏆 Best Match",       top['neighborhood'])
c2.metric("✅ Match Score",      f"{top['Match_Score']:.1f}%")
c3.metric("💵 Est. 2BHK Rent",   f"₹{int(top['Rent_Price']):,}")
c4.metric("🌫️ PM2.5 (Winter)",   f"{top['PM25_Winter']:.1f} µg/m³")
c5.metric("🤖 Cluster Group",    f"Group {int(top['Cluster_ID'])}")

st.divider()

# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT: Map  |  Top Match Profile  |  Rankings
# ══════════════════════════════════════════════════════════════
col_map, col_profile, col_rank = st.columns([1.4, 1.0, 0.9])

# ── MAP ──────────────────────────────────────────────────────
with col_map:
    st.markdown('<p class="section-title">📍 Top 5 on the Map</p>', unsafe_allow_html=True)

    m = folium.Map(
        location=[top['latitude'], top['longitude']],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    colors = ['#22c55e', '#86efac', '#4ade80', '#16a34a', '#15803d']
    for idx, row in top10.head(5).iterrows():
        label = f"#{idx+1} {row['neighborhood']}<br>Score: {row['Match_Score']:.1f}%<br>Rent: ₹{int(row['Rent_Price']):,}"
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10 if idx == 0 else 7,
            color=colors[idx],
            fill=True,
            fill_color=colors[idx],
            fill_opacity=0.85,
            popup=folium.Popup(label, max_width=200),
            tooltip=row['neighborhood']
        ).add_to(m)
        if idx == 0:
            folium.Marker(
                [row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='background:#22c55e;color:white;padding:3px 8px;
                         border-radius:6px;font-weight:700;font-size:11px;
                         white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.4)'>
                        🏆 {row['neighborhood']}
                    </div>""")
            ).add_to(m)

    st_folium(m, height=420, use_container_width=True)

# ── TOP MATCH PROFILE ────────────────────────────────────────
with col_profile:
    st.markdown('<p class="section-title">🔍 Top Match Breakdown</p>', unsafe_allow_html=True)

    # Score badge
    st.markdown(f"""
    <div style='margin-bottom:1rem;'>
        <span style='color:#6b7280;font-size:0.82rem;text-transform:uppercase;
                     letter-spacing:0.08em;'>Match Score</span><br>
        <span class='score-badge'>{top['Match_Score']:.1f}%</span>
        <span style='color:#6b7280;font-size:0.85rem;margin-left:8px;'>
            {top['neighborhood']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Per-factor breakdown bars
    factor_config = [
        ("🍃 Air Quality",  1 - top['PM25_Winter_scaled'],    w_air),
        ("🚗 Low Traffic",  1 - top['Traffic_Index_scaled'],  w_traffic),
        ("💰 Affordability",1 - top['Rent_Price_scaled'],     w_rent),
        ("🎓 Schools",      top['num_schools_scaled'],        w_school),
        ("🏥 Hospitals",    top['num_hospitals_scaled'],      w_hosp),
        ("🚇 Metro",        top['num_metro_stations_scaled'], w_metro),
    ]

    for label, score, weight in factor_config:
        pct = int(score * 100)
        opacity = "1.0" if weight > 0 else "0.35"
        st.markdown(f"""
        <div class='factor-row' style='opacity:{opacity}'>
            <span class='factor-label'>{label}</span>
            <div class='factor-bar-bg'>
                <div class='factor-bar-fill' style='width:{pct}%'></div>
            </div>
            <span class='factor-value'>{pct}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart
    categories = ['Air', 'Traffic', 'Rent', 'Schools', 'Hospitals', 'Metro']
    user_vals  = [w_air, w_traffic, w_rent, w_school, w_hosp, w_metro]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_vals + [user_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(34,197,94,0.15)',
        line=dict(color='#22c55e', width=2),
        name='Your Priorities'
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#111827',
            radialaxis=dict(visible=True, range=[0, 10], color='#374151',
                            gridcolor='#1f2937', tickfont=dict(color='#6b7280', size=9)),
            angularaxis=dict(color='#6b7280', gridcolor='#1f2937',
                             tickfont=dict(color='#9ca3af', size=10))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=240,
        margin=dict(l=30, r=30, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# ── RANKINGS LIST ─────────────────────────────────────────────
with col_rank:
    st.markdown('<p class="section-title">🏅 Top 10 Rankings</p>', unsafe_allow_html=True)

    max_score = top10['Match_Score'].max()
    for idx, row in top10.iterrows():
        score      = row['Match_Score']
        score_color = '#22c55e' if idx == 0 else ('#86efac' if idx < 3 else '#6b7280')
        medal = ['🥇','🥈','🥉'][idx] if idx < 3 else f"#{idx+1}"
        st.markdown(f"""
        <div class='nbhd-card'>
            <div>
                <div class='nbhd-name'>{medal} {row['neighborhood']}</div>
                <div class='nbhd-meta'>₹{int(row['Rent_Price']):,}/mo &nbsp;·&nbsp;
                    PM2.5: {row['PM25_Winter']:.0f} &nbsp;·&nbsp;
                    Cluster {int(row['Cluster_ID'])}</div>
            </div>
            <div class='nbhd-score' style='color:{score_color}'>{score:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2: COMPARISON TABLE  +  RENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════
col_table, col_chart = st.columns([1.3, 1.0])

with col_table:
    st.markdown('<p class="section-title">📋 Top 10 Detailed Comparison</p>', unsafe_allow_html=True)

    display_df = top10[[
        'neighborhood', 'Match_Score', 'Rent_Price',
        'PM25_Winter', 'Traffic_Index',
        'num_schools', 'num_hospitals', 'num_metro_stations',
        'nearest_station'
    ]].copy()

    display_df.columns = [
        'Neighbourhood', 'Score %', 'Rent (₹)',
        'PM2.5', 'Traffic Idx',
        'Schools', 'Hospitals', 'Metro Stns',
        'Nearest Metro'
    ]
    display_df.index = display_df.index + 1
    display_df['Score %']  = display_df['Score %'].round(1)
    display_df['Rent (₹)'] = display_df['Rent (₹)'].apply(lambda x: f"₹{int(x):,}")

    st.dataframe(
        display_df.style.background_gradient(subset=['Score %'], cmap='Greens')
                        .format({'PM2.5': '{:.1f}', 'Traffic Idx': '{:.1f}'}),
        use_container_width=True,
        height=360
    )

with col_chart:
    st.markdown('<p class="section-title">💰 Rent vs Match Score</p>', unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=top10['Rent_Price'],
        y=top10['Match_Score'],
        mode='markers+text',
        text=top10['neighborhood'],
        textposition='top center',
        textfont=dict(color='#9ca3af', size=9, family='DM Sans'),
        marker=dict(
            size=top10['Match_Score'] / 5 + 8,
            color=top10['Match_Score'],
            colorscale=[[0, '#166534'], [1, '#22c55e']],
            showscale=False,
            line=dict(color='#111827', width=1)
        ),
        hovertemplate='<b>%{text}</b><br>Rent: ₹%{x:,.0f}<br>Score: %{y:.1f}%<extra></extra>'
    ))
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#111827',
        height=360,
        xaxis=dict(title='Monthly Rent (₹)', color='#6b7280', gridcolor='#1f2937',
                   tickformat=',.0f', title_font=dict(color='#6b7280')),
        yaxis=dict(title='Match Score (%)', color='#6b7280', gridcolor='#1f2937',
                   title_font=dict(color='#6b7280')),
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family='DM Sans', color='#9ca3af')
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3: FULL NEIGHBOURHOOD EXPLORER
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">🔎 Explore All Neighbourhoods</p>', unsafe_allow_html=True)

search = st.text_input("Search by neighbourhood name", placeholder="e.g. Dwarka, Rohini, Hauz Khas...")
explore_df = ranked_df.copy()
if search:
    explore_df = explore_df[explore_df['neighborhood'].str.contains(search, case=False)]

explore_display = explore_df[[
    'neighborhood', 'Match_Score', 'Rent_Price', 'PM25_Winter',
    'Traffic_Index', 'num_schools', 'num_hospitals',
    'num_metro_stations', 'Cluster_ID'
]].copy()
explore_display.columns = [
    'Neighbourhood', 'Score %', 'Rent (₹)', 'PM2.5',
    'Traffic', 'Schools', 'Hospitals', 'Metro Stns', 'Cluster'
]
explore_display.index = explore_display.index + 1
explore_display['Score %']  = explore_display['Score %'].round(1)
explore_display['Rent (₹)'] = explore_display['Rent (₹)'].apply(lambda x: f"₹{int(x):,}")

st.dataframe(
    explore_display.style.background_gradient(subset=['Score %'], cmap='Greens'),
    use_container_width=True,
    height=300
)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<p style='text-align:center;color:#374151;font-size:0.78rem;'>
    CityMatch &nbsp;·&nbsp; Delhi Neighbourhood Intelligence &nbsp;·&nbsp;
    Data: OSMnx · TomTom · CPCB &nbsp;·&nbsp;
    Built with Streamlit
</p>
""", unsafe_allow_html=True)