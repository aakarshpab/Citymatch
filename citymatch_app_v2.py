import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(
    page_title="CityMatch | Delhi Neighbourhood AI",
    layout="wide",
    page_icon="🏙️",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f7f8fa; }
.main { background-color: #f7f8fa; }
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 1400px; }

[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] * { color: #374151 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #16a34a !important; border: 3px solid #ffffff !important; box-shadow: 0 1px 6px rgba(22,163,74,0.4) !important;
}

h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 1.2rem 1.4rem !important; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
[data-testid="stMetricLabel"] { color: #9ca3af !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stMetricValue"] { color: #111827 !important; font-family: 'Playfair Display', serif !important; font-size: 1.35rem !important; font-weight: 700 !important; }

.section-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #9ca3af; margin-bottom: 1rem; margin-top: 0.2rem; display: block; }

.score-hero { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1.5px solid #bbf7d0; border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; }
.score-number { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 700; color: #15803d; line-height: 1; }
.score-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.score-name { font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 0.4rem; }

.factor-row { display: flex; align-items: center; margin: 9px 0; gap: 12px; }
.factor-label { font-size: 0.8rem; color: #6b7280; width: 105px; flex-shrink: 0; font-weight: 500; }
.factor-bar-bg { flex: 1; background: #f3f4f6; border-radius: 6px; height: 8px; }
.factor-bar-fill { height: 8px; border-radius: 6px; background: linear-gradient(90deg, #16a34a, #22c55e); }
.factor-value { font-size: 0.78rem; color: #374151; font-weight: 600; width: 32px; text-align: right; flex-shrink: 0; }

.nbhd-card { background: #ffffff; border: 1px solid #f3f4f6; border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center; }
.nbhd-card.top-card { border: 1.5px solid #bbf7d0; background: #f0fdf4; }
.nbhd-name { font-size: 0.9rem; font-weight: 600; color: #111827; }
.nbhd-meta { font-size: 0.73rem; color: #9ca3af; margin-top: 2px; }
.nbhd-score { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.1rem; }

hr { border-color: #e5e7eb !important; margin: 2rem 0 !important; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    for fname in ["clustered_results_v5.csv", "clustered_results_v4.csv", "clustered_results_v3.csv"]:
        try:
            df = pd.read_csv(fname)
            features_to_scale = ['PM25_Winter', 'Traffic_Index', 'Rent_Price',
                                  'num_schools', 'num_hospitals', 'num_metro_stations']
            missing = [f for f in features_to_scale if f"{f}_scaled" not in df.columns]
            if missing:
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(df[features_to_scale])
                for i, col in enumerate(features_to_scale):
                    df[f"{col}_scaled"] = scaled[:, i]
            return df
        except FileNotFoundError:
            continue
    return None

df = load_data()
if df is None:
    st.error("Dataset not found. Place clustered_results_v4.csv in the project folder.")
    st.stop()

for outlier in ['Najafgarh', 'Narela', 'Bawana']:
    df = df[df['neighborhood'] != outlier]
df = df.reset_index(drop=True)


def calculate_scores(df, weights):
    df = df.copy()
    df['_air']     = 1 - df['PM25_Winter_scaled']
    df['_traffic'] = 1 - df['Traffic_Index_scaled']
    df['_rent']    = 1 - df['Rent_Price_scaled']
    df['_school']  = df['num_schools_scaled']
    df['_hosp']    = df['num_hospitals_scaled']
    df['_metro']   = df['num_metro_stations_scaled']
    total_weight = sum(weights.values())
    if total_weight == 0:
        df['Match_Score'] = 0.0
    else:
        df['Match_Score'] = (
            weights['air'] * df['_air'] + weights['traffic'] * df['_traffic'] +
            weights['rent'] * df['_rent'] + weights['school'] * df['_school'] +
            weights['hosp'] * df['_hosp'] + weights['metro'] * df['_metro']
        ) / total_weight * 100
    return df.sort_values('Match_Score', ascending=False).reset_index(drop=True)


with st.sidebar:
    st.markdown("<div style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:700;color:#111827;margin-bottom:0.2rem;'>🏙️ CityMatch</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#9ca3af;margin-bottom:1.5rem;'>Delhi Neighbourhood Recommender</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**🎯 Set Your Priorities**")
    st.caption("Slide to 0 to ignore a factor entirely.")
    st.markdown("<br>", unsafe_allow_html=True)
    w_air     = st.slider("🍃 Clean Air",       0, 10, 8)
    st.markdown("<br>", unsafe_allow_html=True)
    w_traffic = st.slider("🚗 Low Traffic",      0, 10, 6)
    st.markdown("<br>", unsafe_allow_html=True)
    w_rent    = st.slider("💰 Affordable Rent",  0, 10, 7)
    st.markdown("<br>", unsafe_allow_html=True)
    w_school  = st.slider("🎓 Schools Nearby",   0, 10, 5)
    st.markdown("<br>", unsafe_allow_html=True)
    w_hosp    = st.slider("🏥 Hospitals Nearby", 0, 10, 5)
    st.markdown("<br>", unsafe_allow_html=True)
    w_metro   = st.slider("🚇 Metro Access",     0, 10, 6)
    st.divider()
    st.caption("Match Score is a normalised weighted sum. Pollution, traffic and rent are inverted so lower always means better.")

weights = {'air': w_air, 'traffic': w_traffic, 'rent': w_rent,
           'school': w_school, 'hosp': w_hosp, 'metro': w_metro}
ranked_df = calculate_scores(df, weights)
top       = ranked_df.iloc[0]
top10     = ranked_df.head(10)


# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.5rem;'>
    <span style='font-family:Playfair Display,serif;font-size:2.2rem;font-weight:700;color:#111827;'>CityMatch</span>
    <span style='font-family:Playfair Display,serif;font-size:2.2rem;font-weight:700;color:#16a34a;'> Delhi</span>
</div>
<p style='font-size:0.9rem;color:#6b7280;margin-top:0;margin-bottom:2rem;'>
    AI-powered neighbourhood matching — adjust the sliders to find your ideal locality
</p>
""", unsafe_allow_html=True)


# ── METRICS ───────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏆 Best Match",     top['neighborhood'])
c2.metric("✅ Match Score",    f"{top['Match_Score']:.1f}%")
c3.metric("💵 Est. 2BHK Rent", f"₹{int(top['Rent_Price']):,}")
c4.metric("🌫️ PM2.5 Winter",   f"{top['PM25_Winter']:.1f} µg/m³")
c5.metric("🤖 Cluster",        f"Group {int(top['Cluster_ID'])}")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 1: MAP | PROFILE ──────────────────────────────────────
col_map, col_gap, col_profile = st.columns([1.5, 0.05, 1.0])

with col_map:
    st.markdown("<span class='section-label'>📍 Top 5 on the Map</span>", unsafe_allow_html=True)

    # OpenStreetMap = standard light/white map tiles
    m = folium.Map(location=[top['latitude'], top['longitude']], zoom_start=11, tiles="OpenStreetMap")

    colors = ['#15803d', '#16a34a', '#22c55e', '#4ade80', '#86efac']
    sizes  = [14, 11, 10, 9, 8]

    for rank, (_, row) in enumerate(top10.head(5).iterrows()):
        popup_html = f"""<div style='font-family:Inter,sans-serif;min-width:160px;padding:4px;'>
            <b style='font-size:13px;color:#111827;'>#{rank+1} {row['neighborhood']}</b><br>
            <span style='color:#6b7280;font-size:11px;'>
                Score: <b style='color:#15803d;'>{row['Match_Score']:.1f}%</b><br>
                Rent: ₹{int(row['Rent_Price']):,}/mo<br>
                PM2.5: {row['PM25_Winter']:.0f} µg/m³<br>
                Nearest Metro: {row['nearest_station']}
            </span></div>"""

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=sizes[rank], color='white', weight=2,
            fill=True, fill_color=colors[rank], fill_opacity=0.92,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"#{rank+1} {row['neighborhood']} — {row['Match_Score']:.0f}%"
        ).add_to(m)

        if rank == 0:
            folium.Marker(
                [row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f"""
                    <div style='background:#15803d;color:white;padding:4px 10px;
                         border-radius:20px;font-weight:700;font-size:11px;
                         font-family:Inter,sans-serif;white-space:nowrap;
                         box-shadow:0 2px 8px rgba(0,0,0,0.2);margin-top:-28px;margin-left:16px;'>
                        🏆 {row['neighborhood']}</div>""")
            ).add_to(m)

    st_folium(m, height=460, use_container_width=True)

with col_profile:
    st.markdown("<span class='section-label'>🔍 Top Match Breakdown</span>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='score-hero'>
        <div class='score-label'>Match Score</div>
        <div class='score-number'>{top['Match_Score']:.1f}%</div>
        <div class='score-name'>{top['neighborhood']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    factor_config = [
        ("🍃 Air Quality",   1 - top['PM25_Winter_scaled'],    w_air),
        ("🚗 Low Traffic",   1 - top['Traffic_Index_scaled'],  w_traffic),
        ("💰 Affordability", 1 - top['Rent_Price_scaled'],     w_rent),
        ("🎓 Schools",       top['num_schools_scaled'],        w_school),
        ("🏥 Hospitals",     top['num_hospitals_scaled'],      w_hosp),
        ("🚇 Metro",         top['num_metro_stations_scaled'], w_metro),
    ]
    for label, score, weight in factor_config:
        pct = int(score * 100)
        opacity = "1.0" if weight > 0 else "0.3"
        st.markdown(f"""
        <div class='factor-row' style='opacity:{opacity};'>
            <span class='factor-label'>{label}</span>
            <div class='factor-bar-bg'><div class='factor-bar-fill' style='width:{pct}%;'></div></div>
            <span class='factor-value'>{pct}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    categories = ['Air', 'Traffic', 'Rent', 'Schools', 'Hospitals', 'Metro']
    user_vals  = [w_air, w_traffic, w_rent, w_school, w_hosp, w_metro]
    fig_radar  = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_vals + [user_vals[0]], theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(22,163,74,0.1)',
        line=dict(color='#16a34a', width=2.5)
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='#fafafa',
            radialaxis=dict(visible=True, range=[0,10], color='#d1d5db',
                            gridcolor='#e5e7eb', tickfont=dict(color='#9ca3af', size=8)),
            angularaxis=dict(color='#9ca3af', gridcolor='#e5e7eb',
                             tickfont=dict(color='#374151', size=10))
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False, height=260, margin=dict(l=35, r=35, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 2: RANKINGS | SCATTER ─────────────────────────────────
col_rank, col_gap2, col_chart = st.columns([1.0, 0.05, 1.2])

with col_rank:
    st.markdown("<span class='section-label'>🏅 Top 10 Rankings</span>", unsafe_allow_html=True)
    for idx, row in top10.iterrows():
        score       = row['Match_Score']
        score_color = '#15803d' if idx == 0 else ('#16a34a' if idx < 3 else '#6b7280')
        medal       = ['🥇','🥈','🥉'][idx] if idx < 3 else f"#{idx+1}"
        card_class  = "nbhd-card top-card" if idx == 0 else "nbhd-card"
        st.markdown(f"""
        <div class='{card_class}'>
            <div>
                <div class='nbhd-name'>{medal}&nbsp; {row['neighborhood']}</div>
                <div class='nbhd-meta'>₹{int(row['Rent_Price']):,}/mo &nbsp;·&nbsp; PM2.5 {row['PM25_Winter']:.0f} &nbsp;·&nbsp; Cluster {int(row['Cluster_ID'])}</div>
            </div>
            <div class='nbhd-score' style='color:{score_color};'>{score:.0f}%</div>
        </div>""", unsafe_allow_html=True)

with col_chart:
    st.markdown("<span class='section-label'>💰 Rent vs Match Score</span>", unsafe_allow_html=True)
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=top10['Rent_Price'], y=top10['Match_Score'],
        mode='markers+text', text=top10['neighborhood'],
        textposition='top center',
        textfont=dict(color='#6b7280', size=9, family='Inter'),
        marker=dict(
            size=top10['Match_Score'] / 6 + 10,
            color=top10['Match_Score'],
            colorscale=[[0,'#bbf7d0'],[0.5,'#22c55e'],[1,'#15803d']],
            showscale=True,
            colorbar=dict(title=dict(text='Score %', font=dict(color='#6b7280',size=10)),
                          tickfont=dict(color='#9ca3af',size=9), thickness=10, len=0.7),
            line=dict(color='white', width=1.5)
        ),
        hovertemplate='<b>%{text}</b><br>Rent: ₹%{x:,.0f}/mo<br>Score: %{y:.1f}%<extra></extra>'
    ))
    fig_sc.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#fafafa', height=430,
        xaxis=dict(title='Monthly Rent (₹)', color='#9ca3af', gridcolor='#f3f4f6',
                   tickformat=',.0f', title_font=dict(color='#6b7280',size=11),
                   showline=True, linecolor='#e5e7eb'),
        yaxis=dict(title='Match Score (%)', color='#9ca3af', gridcolor='#f3f4f6',
                   title_font=dict(color='#6b7280',size=11),
                   showline=True, linecolor='#e5e7eb'),
        margin=dict(l=20, r=20, t=20, b=20), font=dict(family='Inter', color='#374151')
    )
    st.plotly_chart(fig_sc, use_container_width=True)


st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 3: DETAILED TABLE ─────────────────────────────────────
st.markdown("<span class='section-label'>📋 Top 10 Detailed Comparison</span>", unsafe_allow_html=True)

display_df = top10[['neighborhood','Match_Score','Rent_Price','PM25_Winter',
                     'Traffic_Index','num_schools','num_hospitals',
                     'num_metro_stations','nearest_station']].copy()
display_df.columns = ['Neighbourhood','Score %','Rent (₹)','PM2.5','Traffic Idx',
                      'Schools','Hospitals','Metro Stns','Nearest Metro']
display_df.index        = display_df.index + 1
display_df['Score %']   = display_df['Score %'].round(1)
display_df['Rent (₹)']  = display_df['Rent (₹)'].apply(lambda x: f"₹{int(x):,}")

st.dataframe(
    display_df.style.background_gradient(subset=['Score %'], cmap='Greens')
                    .format({'PM2.5':'{:.1f}','Traffic Idx':'{:.2f}'}),
    use_container_width=True, height=380
)


st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 4: EXPLORER ───────────────────────────────────────────
st.markdown("<span class='section-label'>🔎 Explore All Neighbourhoods</span>", unsafe_allow_html=True)
search = st.text_input("Search", placeholder="Type a neighbourhood — e.g. Dwarka, Rohini, Hauz Khas...", label_visibility="collapsed")
st.markdown("<br>", unsafe_allow_html=True)

explore_df = ranked_df.copy()
if search:
    explore_df = explore_df[explore_df['neighborhood'].str.contains(search, case=False, na=False)]

exp = explore_df[['neighborhood','Match_Score','Rent_Price','PM25_Winter',
                   'Traffic_Index','num_schools','num_hospitals',
                   'num_metro_stations','Cluster_ID']].copy()
exp.columns = ['Neighbourhood','Score %','Rent (₹)','PM2.5','Traffic','Schools','Hospitals','Metro Stns','Cluster']
exp.index        = exp.index + 1
exp['Score %']   = exp['Score %'].round(1)
exp['Rent (₹)']  = exp['Rent (₹)'].apply(lambda x: f"₹{int(x):,}")

st.dataframe(exp.style.background_gradient(subset=['Score %'], cmap='Greens'),
             use_container_width=True, height=340)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("""
<p style='text-align:center;color:#d1d5db;font-size:0.75rem;font-family:Inter,sans-serif;letter-spacing:0.05em;'>
    CityMatch &nbsp;·&nbsp; Delhi Neighbourhood Intelligence &nbsp;·&nbsp;
    Data: OSMnx · TomTom API · CPCB · Kaggle &nbsp;·&nbsp; Built with Streamlit
</p>""", unsafe_allow_html=True)