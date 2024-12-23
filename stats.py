import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Load the log data into a pandas DataFrame
log_file_path = '13-48_23-12_key_log.txt'

def load_log_data(file_path):
    # Read the log file into a DataFrame
    data = pd.read_csv(file_path, sep=" - ", header=None, names=["timestamp", "coords", "action", "eyeblink_count"], engine='python')

    # Remove the square brackets around the timestamp and convert to datetime
    data["timestamp"] = data["timestamp"].str.strip("[]")  # Remove the square brackets
    data["timestamp"] = pd.to_datetime(data["timestamp"])  # Convert timestamps to datetime objects

    # Ensure eyeblink_count is an integer
    data["eyeblink_count"] = data["eyeblink_count"].astype(int)

    return data

# Load data
data = load_log_data(log_file_path)

# 1. Eyeblink Count Over Time
def plot_eyeblink_count_over_time(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["eyeblink_count"], label="Eyeblink Count")
    plt.title("Eyeblink Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_eyeblink_count_over_time(data)

# 2. Most Frequent Actions
def plot_most_frequent_actions(data):
    action_counts = data['action'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values)
    plt.title("Most Frequent Actions")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

plot_most_frequent_actions(data)

# 3. Action Sequences and Combinations
def plot_action_sequences(data, sequence_length=2):
    sequences = data['action'].shift(-sequence_length).dropna().astype(str) + ' -> ' + data['action'].astype(str)
    sequence_counts = sequences.value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sequence_counts.index, y=sequence_counts.values)
    plt.title(f"Most Frequent {sequence_length}-Action Sequences")
    plt.xlabel(f"Action Sequences ({sequence_length}-actions)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_action_sequences(data, sequence_length=2)  # You can change the sequence length

# 4. Action Duration (Time between actions)
def plot_action_duration(data):
    data['time_diff'] = data['timestamp'].diff().dt.total_seconds()
    action_durations = data.groupby('action')['time_diff'].mean()
    plt.figure(figsize=(10, 6))
    action_durations.plot(kind='bar')
    plt.title("Average Action Duration")
    plt.xlabel("Action")
    plt.ylabel("Average Duration (seconds)")
    plt.tight_layout()
    plt.show()

plot_action_duration(data)

# 5. Action vs Eyeblink Count
def plot_action_vs_eyeblink_count(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="action", y="eyeblink_count", data=data)
    plt.title("Action vs Eyeblink Count")
    plt.xlabel("Action")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_action_vs_eyeblink_count(data)

# 6. Action Heatmap (if coordinates are useful)
def plot_action_heatmap(data):
    # Assume coords are (x, y) and can be split into two columns (this may need to be adapted)
    data[['x', 'y']] = data['coords'].str.extract(r'(\d+),\s*(\d+)').astype(float)
    plt.figure(figsize=(10, 6))
    plt.hexbin(data['x'], data['y'], gridsize=30, cmap='YlGnBu')
    plt.colorbar(label="Frequency")
    plt.title("Action Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()

plot_action_heatmap(data)

# 7. Correlation Between Actions and Eyeblink Counts (Statistical Analysis)
def action_eyeblink_correlation(data):
    action_dummies = pd.get_dummies(data['action'])
    action_dummies['eyeblink_count'] = data['eyeblink_count']
    correlation = action_dummies.corr()['eyeblink_count'].sort_values(ascending=False)
    print(correlation)

action_eyeblink_correlation(data)

# 8. Long-Term Trends in User Behavior (Change in Action Frequency Over Time)
def plot_action_trends_over_time(data):
    data['hour'] = data['timestamp'].dt.hour
    action_trends = data.groupby(['hour', 'action']).size().unstack(fill_value=0)
    action_trends.plot(kind='line', stacked=True, figsize=(12, 8))
    plt.title("Action Frequency by Hour of the Day")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Action Frequency")
    plt.tight_layout()
    plt.show()

plot_action_trends_over_time(data)

# 9. Action Timing and Sequence Patterns
def action_timing_analysis(data):
    data['time_diff'] = data['timestamp'].diff().dt.total_seconds()
    action_timing = data.groupby('action')['time_diff'].mean()
    print(action_timing)

action_timing_analysis(data)
