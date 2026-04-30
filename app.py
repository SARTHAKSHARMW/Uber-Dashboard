# =========================================
# 🚀 ULTIMATE UBER DASHBOARD (TOP 1%)
# =========================================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="Uber AI Dashboard", layout="wide")

# -----------------------------------------
# CUSTOM CSS (🔥 PRO UI)
# -----------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.title {
    font-size: 30px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("uber.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# -----------------------------------------
# DATETIME
# -----------------------------------------
if 'date/time' in df.columns:
    df['datetime'] = pd.to_datetime(df['date/time'])
else:
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day_name()
df['month'] = df['datetime'].dt.month

# -----------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Analysis", "Map", "Prediction"])

st.sidebar.markdown("---")
days = st.sidebar.multiselect("Day", df['day'].unique(), default=df['day'].unique())
months = st.sidebar.multiselect("Month", df['month'].unique(), default=df['month'].unique())

filtered_df = df[(df['day'].isin(days)) & (df['month'].isin(months))]

# =========================================
# PAGE 1: DASHBOARD
# =========================================
if page == "Dashboard":

    st.title("🚖 Uber AI Dashboard")
    st.markdown("### Real-Time Demand Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div class='card'><p>Total Trips</p><h2>{len(filtered_df)}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><p>Peak Hour</p><h2>{filtered_df['hour'].mode()[0]}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><p>Peak Day</p><h2>{filtered_df['day'].mode()[0]}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'><p>Months</p><h2>{len(months)}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    fig = px.histogram(filtered_df, x="hour", color_discrete_sequence=["#3b82f6"])
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE 2: ANALYSIS
# =========================================
elif page == "Analysis":

    st.title("📊 Advanced Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(filtered_df, x="day", color_discrete_sequence=["#10b981"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pivot = filtered_df.groupby(['day','hour']).size().reset_index(name='count')
        fig = px.density_heatmap(pivot, x="hour", y="day", z="count")
        st.plotly_chart(fig, use_container_width=True)

# =========================================
# PAGE 3: MAP
# =========================================
elif page == "Map":

    st.title("📍 Location Intelligence")

    lat_col, lon_col = None, None
    for col in df.columns:
        if 'lat' in col:
            lat_col = col
        if 'lon' in col:
            lon_col = col

    if lat_col and lon_col:
        st.map(filtered_df[[lat_col, lon_col]])
    else:
        st.warning("No location data available")

# =========================================
# PAGE 4: ML PREDICTION
# =========================================
elif page == "Prediction":

    st.title("🤖 AI Prediction Engine")

    hourly_data = df.groupby(['month','day','hour']).size().reset_index(name='rides')
    hourly_data['day_num'] = hourly_data['day'].astype('category').cat.codes

    X = hourly_data[['month','day_num','hour']]
    y = hourly_data['rides']

    model = RandomForestRegressor()
    model.fit(X, y)

    col1, col2, col3 = st.columns(3)

    h = col1.slider("Hour", 0, 23, 12)
    d = col2.selectbox("Day", df['day'].unique())
    m = col3.selectbox("Month", df['month'].unique())

    day_num = df['day'].astype('category').cat.codes[df['day'] == d].iloc[0]

    pred = model.predict([[m, day_num, h]])

    st.success(f"🚀 Predicted Demand: {int(pred[0])}")

    st.markdown("---")

    st.subheader("Feature Importance")
    fig = px.bar(x=X.columns, y=model.feature_importances_)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# DOWNLOAD
# -----------------------------------------
st.sidebar.markdown("---")
st.sidebar.download_button("⬇️ Download Data", df.to_csv(index=False), "uber.csv")