import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from matplotlib.animation import FuncAnimation
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#  Anomaly Detector class
class AnomalyDetector:
    def __init__(self, contamination=0.01, window_size=100, alpha=0.1):
        # Initialize the Isolation Forest model
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.window_size = window_size
        # Use deques for efficient sliding window operations
        self.data = deque(maxlen=self.window_size)
        self.scores = deque(maxlen=self.window_size)
        self.alpha = alpha  # For exponential moving average
        self.ema = None
        self.threshold = None
        self.update_frequency = 10  # Update model every 10 points

    def is_anomaly(self, value):
        self.data.append(value)

        # Update Exponential Moving Average (EMA)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        if len(self.data) < self.window_size:
            return False, 0, self.ema  # Wait until window is full

        # Update model periodically
        if len(self.data) % self.update_frequency == 0:
            data_array = np.array(self.data).reshape(-1, 1)
            self.model.fit(data_array)

        # Get anomaly score
        score = -self.model.score_samples([[value]])[0]
        self.scores.append(score)

        # Update adaptive threshold
        if self.threshold is None:
            self.threshold = np.mean(self.scores) + 2 * np.std(self.scores)
        else:
            self.threshold = 0.9 * self.threshold + 0.1 * (np.mean(self.scores) + 2 * np.std(self.scores))

        is_anomaly = score > self.threshold
        return is_anomaly, score, self.ema

# Function to generate complex data with trends, seasonality, and anomalies
def generate_complex_data():
    t = 0
    trend = 0
    while True:
        trend += np.random.normal(0, 0.01)
        seasonality = 5 * np.sin(0.1 * t) + 3 * np.cos(0.05 * t)
        noise = np.random.normal(0, 1)
        value = 100 + trend + seasonality + noise
        anomaly = 0

        # Introduce anomalies with 2% probability
        if np.random.random() < 0.02:
            anomaly_type = np.random.choice(['spike', 'level_shift'])
            anomaly = 1
            if anomaly_type == 'spike':
                value += np.random.choice([-1, 1]) * np.random.uniform(15, 25)
            else:
                trend += np.random.choice([-1, 1]) * np.random.uniform(5, 10)

        yield t, value, anomaly
        t += 1

# Function to update the plot for each animation frame
def update_plot(frame):
    global detector, y_true, y_pred, scores, emas
    t, value, ground_truth_anomaly = next(data_generator)
    times.append(t)
    values.append(value)
    y_true.append(ground_truth_anomaly)

    # Detect anomalies
    is_anomaly, score, ema = detector.is_anomaly(value)
    y_pred.append(1 if is_anomaly else 0)
    anomalies.append(value if is_anomaly else np.nan)
    scores.append(score)
    emas.append(ema)

    # Limit the number of points shown to 500
    if len(times) > 500:
        times.pop(0)
        values.pop(0)
        anomalies.pop(0)
        scores.pop(0)
        emas.pop(0)

    # Update plot data
    line.set_data(times, values)
    scatter.set_offsets(np.c_[times, anomalies])
    ema_line.set_data(times, emas)

    # Adjust plot limits
    ax.relim()
    ax.autoscale_view()

    # Calculate performance metrics
    if len(y_true) >= detector.window_size:
        precision = precision_score(y_true[-detector.window_size:], y_pred[-detector.window_size:])
        recall = recall_score(y_true[-detector.window_size:], y_pred[-detector.window_size:])
        f1 = f1_score(y_true[-detector.window_size:], y_pred[-detector.window_size:])
        accuracy = accuracy_score(y_true[-detector.window_size:], y_pred[-detector.window_size:])
        false_positive_rate = sum([1 for i, j in zip(y_true[-detector.window_size:], y_pred[-detector.window_size:]) if
                                   j == 1 and i == 0]) / detector.window_size
    else:
        precision, recall, f1, accuracy, false_positive_rate = 0, 0, 0, 0, 0

    # Update statistics text
    stats_text.set_text(
        "Mean: {:.2f}\n"
        "Std Dev: {:.2f}\n"
        "Precision: {:.2f}\n"
        "Recall: {:.2f}\n"
        "F1-Score: {:.2f}\n"
        "Accuracy: {:.2f}\n"
        "False Positive Rate: {:.2f}\n"
        "Anomalies: {}".format(
            np.mean(values), np.std(values), precision, recall, f1, accuracy,
            false_positive_rate, np.sum(~np.isnan(anomalies)))
    )

    return line, scatter, ema_line, stats_text

# Set up the plot
fig, ax = plt.subplots(figsize=(15, 8))
line, = ax.plot([], [], lw=2, label='Data Stream')
scatter = ax.scatter([], [], color='red', zorder=5, label='Anomalies')
ema_line, = ax.plot([], [], color='green', lw=2, label='Moving Average')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title(' Real-time Adaptive Anomaly Detection')
ax.legend()

# Add statistics text box
stats_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Initialize data structures
times, values, anomalies, scores, emas = [], [], [], [], []
y_true, y_pred = [], []
data_generator = generate_complex_data()
detector = AnomalyDetector()

# Create and display the animation
anim = FuncAnimation(fig, update_plot, frames=1000, interval=50, blit=True)
plt.tight_layout()
plt.show()