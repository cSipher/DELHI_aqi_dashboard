# ================================
# DELHI AQI EARLY WARNING DASHBOARD
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Delhi AQI Early Warning Dashboard",
    layout="wide"
)

st.title("🌫 Delhi AQI Early Warning Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/delhi_ncr_aqi_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df

df = load_data()

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df["aqi_lag1"] = df["aqi"].shift(1)
df["aqi_lag3"] = df["aqi"].shift(3)
df["aqi_roll3"] = df["aqi"].rolling(3).mean()
df["aqi_roll7"] = df["aqi"].rolling(7).mean()
df["aqi_delta1"] = df["aqi"] - df["aqi_lag1"]
df["aqi_delta3"] = df["aqi"] - df["aqi_lag3"]

df = df.dropna()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["datetime"].min().date(), df["datetime"].max().date()]
)

stations = st.sidebar.multiselect(
    "Select Station",
    df["station"].unique(),
    default=df["station"].unique()[:5]
)

filtered_df = df[
    (df["datetime"].dt.date >= date_range[0]) &
    (df["datetime"].dt.date <= date_range[1]) &
    (df["station"].isin(stations))
].copy()

# -----------------------------
# KEY METRICS
# -----------------------------
st.subheader("📊 Key AQI Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Average AQI", round(filtered_df["aqi"].mean(), 2))
c2.metric("Max AQI", int(filtered_df["aqi"].max()))
c3.metric("Severe Days %", round((filtered_df["aqi"] > 400).mean() * 100, 2))

# -----------------------------
# AQI TREND
# -----------------------------
st.subheader("📈 AQI Trend Over Time")

daily = filtered_df.groupby(filtered_df["datetime"].dt.date)["aqi"].mean().reset_index()
daily.columns = ["date", "aqi"]
st.plotly_chart(px.line(daily, x="date", y="aqi"), use_container_width=True)

# -----------------------------
# STATION COMPARISON
# -----------------------------
st.subheader("📍 Station-wise AQI")

station_avg = filtered_df.groupby("station")["aqi"].mean().reset_index()
st.plotly_chart(px.bar(station_avg, x="station", y="aqi"), use_container_width=True)

# -----------------------------
# ML MODEL
# -----------------------------
st.subheader("🤖 ML Severe AQI Early Warning")

feature_cols = [
    "pm25","pm10","no2","so2","co","o3",
    "temperature","humidity","wind_speed","visibility",
    "aqi_lag1","aqi_lag3","aqi_roll3","aqi_roll7",
    "aqi_delta1","aqi_delta3"
]

model = joblib.load("model/xgb_aqi_model.pkl")

X = filtered_df[feature_cols]
filtered_df["severe_prob"] = model.predict_proba(X)[:, 1]

filtered_df["alert_level"] = pd.cut(
    filtered_df["severe_prob"],
    bins=[0, 0.3, 0.6, 0.85, 1],
    labels=["NORMAL", "WARNING", "HIGH ALERT", "EMERGENCY"]
)

st.success("ML Prediction Running Successfully")

# -----------------------------
# ALERTS
# -----------------------------
st.subheader("🚨 Alert Distribution")
st.write(filtered_df["alert_level"].value_counts())

st.subheader("⚠ Latest Alerts")
st.dataframe(
    filtered_df[
        ["datetime", "station", "aqi", "severe_prob", "alert_level"]
    ].sort_values("datetime", ascending=False).head(20)
)

# -----------------------------
# STATION RISK RANKING
# -----------------------------
st.markdown("## 🚨 Station Risk Ranking (Future Severe AQI Risk)")

station_risk = (
    filtered_df
    .groupby("station")
    .agg(
        avg_severe_prob=("severe_prob", "mean"),
        emergency_rate=("alert_level", lambda x: (x == "EMERGENCY").mean() * 100),
        total_records=("severe_prob", "count")
    )
    .reset_index()
    .sort_values("avg_severe_prob", ascending=False)
)

station_risk["Rank"] = range(1, len(station_risk) + 1)

st.dataframe(
    station_risk[["Rank","station","avg_severe_prob","emergency_rate","total_records"]],
    use_container_width=True
)

# -----------------------------
# 24-HOUR FORECAST
# -----------------------------
st.markdown("## 🔮 Next 24 Hours Severe AQI Risk Probability")

latest_df = filtered_df.sort_values("datetime").groupby("station").tail(1)

forecast_rows = []
for _, row in latest_df.iterrows():
    for h in range(24):
        forecast_rows.append({
            "datetime": row["datetime"] + pd.Timedelta(hours=h + 1),
            "station": row["station"],
            "severe_prob": row["severe_prob"]
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df["lower"] = (forecast_df["severe_prob"] * 0.95).clip(0, 1)
forecast_df["upper"] = (forecast_df["severe_prob"] * 1.05).clip(0, 1)

fig = px.line(
    forecast_df,
    x="datetime",
    y="severe_prob",
    color="station"
)

# Thresholds
fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
              annotation_text="⚠ WARNING")
fig.add_hline(y=0.85, line_dash="dash", line_color="red",
              annotation_text="🚨 HIGH ALERT")

fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SHAP EXPLAINABILITY
# -----------------------------
st.markdown("## 🧠 SHAP Explainability (Model Transparency)")

X_shap = filtered_df[feature_cols].sample(300, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

st.subheader("🌍 Global Feature Importance")

fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
st.pyplot(fig1)
plt.clf()

st.subheader("🧩 Feature Impact Distribution")

fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values, X_shap, show=False)
st.pyplot(fig2)
plt.clf()

# -----------------------------
# RAW DATA
# -----------------------------
st.subheader("📂 View Raw Data")
st.dataframe(filtered_df.head(100))
