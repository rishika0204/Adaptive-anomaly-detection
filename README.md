# Real-time Adaptive Anomaly Detection System

This project implements a **real-time adaptive anomaly detection system** using **Isolation Forest**. The system detects anomalies in a stream of data and adapts its detection threshold based on recent data trends, making it suitable for dynamic environments.

## Key Features

- **Real-time Detection**: Continuously monitors data and flags anomalies as they occur.
- **Adaptive Threshold**: Uses an **Exponential Moving Average (EMA)** and dynamically adjusts the anomaly threshold based on recent data.
- **Sliding Window**: A sliding window approach ensures efficient processing and model updates.
- **Complex Data Simulation**: Generates data with trends, seasonality, and noise. Introduces random anomalies like spikes or level shifts.
- **Performance Metrics**: Calculates precision, recall, F1-score, accuracy, and false positive rate during real-time detection.
- **Visualization**: Displays real-time data, anomalies, and moving averages with live performance metrics.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn

Install the necessary libraries using the following command:

```bash
pip install numpy matplotlib scikit-learn
```

## How It Works

### 1. **Anomaly Detector Class**
The `AnomalyDetector` class uses the **Isolation Forest** algorithm to detect anomalies. It maintains a sliding window of recent data and periodically updates the Isolation Forest model.

- **Initialization**:
  - `contamination`: Proportion of anomalies expected in the data.
  - `window_size`: Number of data points stored in the sliding window.
  - `alpha`: Smoothing factor for the Exponential Moving Average.

- **Anomaly Detection**:
  The system calculates an anomaly score for each new data point and compares it to an adaptive threshold to determine if the point is an anomaly.

### 2. **Complex Data Generation**
The `generate_complex_data()` function simulates data with the following components:
- **Trend**: A slow drift over time.
- **Seasonality**: A repeating sinusoidal pattern.
- **Noise**: Random fluctuations.
- **Anomalies**: Occur with a 2% probability, taking the form of either sudden spikes or level shifts.

### 3. **Real-time Visualization**
The real-time plot displays:
- **Data Stream**: The raw values being analyzed.
- **Anomalies**: Marked with red dots when detected.
- **EMA Line**: The Exponential Moving Average of the data.

Additionally, the plot shows live performance metrics like precision, recall, F1-score, accuracy, and false positive rate, which are updated as new data points are processed.

### 4. **Performance Metrics**
The following metrics are calculated in real time over the sliding window:
- **Precision**: Fraction of detected anomalies that are true anomalies.
- **Recall**: Fraction of true anomalies that are detected.
- **F1-Score**: Harmonic mean of precision and recall.
- **Accuracy**: Fraction of correctly classified points.
- **False Positive Rate**: Proportion of normal points incorrectly flagged as anomalies.

## How to Run the Project

1. Clone the repository or copy the code to your local machine.
2. Install the required dependencies using the command provided above.
3. Run the script to launch the real-time anomaly detection system and visualize the results:

```bash
python anomaly_detection.py
```

The real-time plot will display data as it streams in, marking anomalies and updating the moving average and performance metrics.

