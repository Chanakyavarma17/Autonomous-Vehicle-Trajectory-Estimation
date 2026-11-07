
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Autonomous Vehicle | Kalman Filter",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Autonomous Vehicle Trajectory Estimation")
st.caption("GPS/IMU Sensor Fusion | Kalman Filtering | KITTI Dataset")

base = os.path.dirname(__file__)
results = os.path.join(base, "results")

data_file = os.path.join(results, "kalman_estimated_trajectory.csv")
metrics_file = os.path.join(results, "evaluation_metrics.csv")
synthetic_file = os.path.join(results, "synthetic_trajectory.csv")

if not os.path.exists(data_file):
    st.error("Processed trajectory data is missing.")
    st.stop()

df = pd.read_csv(data_file)

required = ["x_gps", "y_gps", "x_estimated", "y_estimated"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"Missing trajectory columns: {missing}")
    st.stop()

st.sidebar.header("Dashboard Controls")

sample_count = st.sidebar.slider(
    "Number of samples",
    min_value=10,
    max_value=len(df),
    value=len(df)
)

data = df.iloc[:sample_count].copy()

# Calculate position error
position_error = np.sqrt(
    (data["x_gps"] - data["x_estimated"])**2 +
    (data["y_gps"] - data["y_estimated"])**2
)

position_rmse = np.sqrt(np.mean(position_error**2))

# Velocity metrics, when available
velocity_rmse = None

if all(c in data.columns for c in [
    "vx_measured", "vy_measured",
    "vx_estimated", "vy_estimated"
]):
    velocity_error = np.sqrt(
        (data["vx_measured"] - data["vx_estimated"])**2 +
        (data["vy_measured"] - data["vy_estimated"])**2
    )
    velocity_rmse = np.sqrt(np.mean(velocity_error**2))

# Summary metrics
st.subheader("Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Samples", f"{len(data):,}")
c2.metric("Position RMSE", f"{position_rmse:.3f} m")

if velocity_rmse is not None:
    c3.metric("Velocity RMSE", f"{velocity_rmse:.3f} m/s")
else:
    c3.metric("Velocity RMSE", "Not available")

if all(c in data.columns for c in [
    "vx_estimated", "vy_estimated"
]):
    speed = np.sqrt(
        data["vx_estimated"].iloc[-1]**2 +
        data["vy_estimated"].iloc[-1]**2
    )
    c4.metric("Final Speed", f"{speed:.2f} m/s")
else:
    c4.metric("Final Speed", "Not available")

st.divider()

# Trajectory chart
st.subheader("1. Vehicle Trajectory")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["x_gps"], y=data["y_gps"],
    mode="lines", name="Observed GPS"
))

fig.add_trace(go.Scatter(
    x=data["x_estimated"], y=data["y_estimated"],
    mode="lines", name="Kalman Estimate"
))

fig.update_layout(
    xaxis_title="East-West Position (m)",
    yaxis_title="North-South Position (m)",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Position error
st.subheader("2. Position Error Over Time")

time_values = (
    data["time"] if "time" in data.columns
    else np.arange(len(data))
)

error_fig = go.Figure()

error_fig.add_trace(go.Scatter(
    x=time_values,
    y=position_error,
    mode="lines",
    name="Position Error"
))

error_fig.add_hline(
    y=position_rmse,
    line_dash="dash",
    annotation_text=f"RMSE: {position_rmse:.3f} m"
)

error_fig.update_layout(
    xaxis_title="Time (seconds)",
    yaxis_title="Position Error (m)",
    template="plotly_white",
    height=400
)

st.plotly_chart(error_fig, use_container_width=True)

# Velocity chart
st.subheader("3. Vehicle Velocity")

if all(c in data.columns for c in [
    "vx_estimated", "vy_estimated"
]):
    velocity_fig = go.Figure()

    velocity_fig.add_trace(go.Scatter(
        x=time_values,
        y=data["vx_estimated"],
        name="Eastward Velocity"
    ))

    velocity_fig.add_trace(go.Scatter(
        x=time_values,
        y=data["vy_estimated"],
        name="Northward Velocity"
    ))

    velocity_fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Velocity (m/s)",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(velocity_fig, use_container_width=True)
else:
    st.info("Velocity columns are not available.")

# Synthetic experiment
st.subheader("4. Synthetic GPS Noise Experiment")

if os.path.exists(synthetic_file):
    syn = pd.read_csv(synthetic_file)

    if all(c in syn.columns for c in [
        "original_x", "original_y",
        "noisy_x", "noisy_y",
        "filtered_x", "filtered_y"
    ]):
        syn_fig = go.Figure()

        syn_fig.add_trace(go.Scatter(
            x=syn["original_x"], y=syn["original_y"],
            mode="lines", name="Original KITTI GPS"
        ))

        syn_fig.add_trace(go.Scatter(
            x=syn["noisy_x"], y=syn["noisy_y"],
            mode="markers", name="Synthetic Noisy GPS"
        ))

        syn_fig.add_trace(go.Scatter(
            x=syn["filtered_x"], y=syn["filtered_y"],
            mode="lines", name="Filtered Estimate"
        ))

        syn_fig.update_layout(
            xaxis_title="East-West Position (m)",
            yaxis_title="North-South Position (m)",
            template="plotly_white",
            height=450
        )

        st.plotly_chart(syn_fig, use_container_width=True)
else:
    st.info("Synthetic experiment results are not available.")

# Recorded metrics
st.subheader("5. Evaluation Results")

if os.path.exists(metrics_file):
    metrics = pd.read_csv(metrics_file)
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("Evaluation metrics file not found.")

# Data table and download
st.subheader("6. Trajectory Data")

st.dataframe(data.head(100), use_container_width=True)

st.download_button(
    "Download trajectory CSV",
    data=df.to_csv(index=False),
    file_name="kalman_estimated_trajectory.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully.")
