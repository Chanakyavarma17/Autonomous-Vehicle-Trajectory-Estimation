# Autonomous Vehicle Trajectory Estimation Using Kalman Filtering

## Project Overview
This project estimates vehicle position and velocity using GPS/IMU sensor fusion and Kalman Filtering.

## Objectives
- Process KITTI Raw GPS/IMU measurements.
- Build a linear state-space model.
- Implement a Kalman Filter.
- Estimate vehicle position and velocity.
- Visualize observed and estimated trajectories.
- Evaluate estimation performance.

## Dataset
KITTI Raw Dataset: 2011_09_26_drive_0009_sync.
Dataset files are excluded from this repository.

## Technologies
- Python, NumPy, Pandas
- Matplotlib, Plotly
- Streamlit
- Jupyter Notebook

## Workflow
1. Dataset collection
2. Preprocessing
3. State-space model formulation
4. Kalman Filter implementation
5. Position and velocity estimation
6. Performance evaluation
7. Visualization

## Dashboard
The dashboard displays vehicle trajectories, GPS and Kalman estimates, velocity analysis and performance metrics.

## Run Dashboard
Install dependencies and run: `streamlit run app.py`

## Disclaimer
This is an educational prototype.