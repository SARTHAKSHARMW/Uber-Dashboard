# =========================================
# 🚀 ULTIMATE UBER DASHBOARD (PREMIUM UI)
# =========================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="Uber AI Dashboard", layout="wide")

# -----------------------------------------
# PREMIUM CSS (🔥 UI UPGRADE)
# -----------------------------------------
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("uber_small.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

with st.spinner("🚀 Loading Uber Data..."):
    df = load_data()

# -----------------------------------------
# DATETIME
# -----------------------------------------
df['datetime'] = pd.to_datetime(df.iloc[:,0])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()
df['month'] = df['datetime'].dt.month

# -----------------------------------------
# SIDEBAR
# -----------------------------------------
st.sidebar.title("🔍 Filters")

days = st.sidebar.multiselect("Day", df['day'].unique(), default=df['day'].unique())
months = st.sidebar.multiselect("Month", df['month'].unique(), default=df['month'].unique())

filtered_df = df[(df['day'].isin(days)) & (df['month'].isin(months))]

# -----------------------------------------
# HEADER
# -----------------------------------------
st.title("🚖 Uber Demand Intelligence")
st.markdown("### Premium Analytics Dashboard")

# -----------------------------------------
# KPI CARDS
# -----------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"<div class='card'><h3>Total Trips</h3><h2>{len(filtered_df)}</h2></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><h3>Peak Hour</h3><h2>{filtered_df['hour'].mode()[0]}</h2></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><h3>Peak Day</h3><h2>{filtered_df['day'].mode()[0]}</h2></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='card'><h3>Months</h3><h2>{len(months)}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------
# TABS
# -----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "🔥 Heatmap", "🤖 Prediction", "📌 Insights"])

# =========================================
# TRENDS
# =========================================
with tab1:
    st.subheader("Trips by Hour")
    fig1 = px.histogram(filtered_df, x="hour", color_discrete_sequence=["#3b82f6"])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Trips by Day")
    fig2 = px.histogram(filtered_df, x="day", color_discrete_sequence=["#10b981"])
    st.plotly_chart(fig2, use_container_width=True)

# =========================================
# HEATMAP (🔥 NEW)
# =========================================
with tab2:
    st.subheader("Uber Pickup Heatmap")

    lat_col, lon_col = None, None
    for col in df.columns:
        if 'lat' in col:
            lat_col = col
        if 'lon' in col:
            lon_col = col

    if lat_col and lon_col:
        m = folium.Map(location=[filtered_df[lat_col].mean(), filtered_df[lon_col].mean()], zoom_start=10)

        heat_data = list(zip(filtered_df[lat_col], filtered_df[lon_col]))
        HeatMap(heat_data).add_to(m)

        st_folium(m, width=700, height=500)
    else:
        st.warning("No location data available")

# =========================================
# ML
# =========================================
with tab3:
    st.subheader("Demand Prediction")

    hourly = df.groupby(['month','day','hour']).size().reset_index(name='rides')
    hourly['day_num'] = hourly['day'].astype('category').cat.codes

    X = hourly[['month','day_num','hour']]
    y = hourly['rides']

    model = RandomForestRegressor()
    model.fit(X, y)

    c1, c2, c3 = st.columns(3)

    h = c1.slider("Hour", 0, 23, 12)
    d = c2.selectbox("Day", df['day'].unique())
    m = c3.selectbox("Month", df['month'].unique())

    day_num = df['day'].astype('category').cat.codes[df['day'] == d].iloc[0]

    pred = model.predict([[m, day_num, h]])

    st.success(f"🚀 Predicted Demand: {int(pred[0])}")

# =========================================
# INSIGHTS
# =========================================
with tab4:
    st.markdown("""
    ### 📊 Key Insights
    - Evening hours have peak demand  
    - Weekends are busiest  
    - Certain zones act as hotspots  
    - Surge pricing likely during peaks  
    """)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.markdown("🚀 Built by Sarthak | Uber AI Dashboard")
