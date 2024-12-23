import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ---------------------------------------
# 1. Load the log data (Keyboard/Mouse)
# ---------------------------------------
def load_log_data(file_path):
    """
    Loads keyboard/mouse log data from a specified file path
    and returns it as a pandas DataFrame.
    """
    data = pd.read_csv(file_path, sep=" - ", header=None, 
                       names=["timestamp", "coords", "action", "eyeblink_count"], 
                       engine='python')
    # Remove the square brackets around the timestamp and convert to datetime
    data["timestamp"] = data["timestamp"].str.strip("[]")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Ensure eyeblink_count is an integer
    data["eyeblink_count"] = data["eyeblink_count"].astype(int)

    return data


# ---------------------------------------
# 2. Basic Plots (as you already have)
# ---------------------------------------
def plot_eyeblink_count_over_time(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["eyeblink_count"], label="Eyeblink Count")
    plt.title("Eyeblink Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_most_frequent_actions(data):
    action_counts = data['action'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values)
    plt.title("Most Frequent Actions")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_action_sequences(data, sequence_length=2):
    """
    Instead of the naive approach, we can build pairs from consecutive rows.
    shift(-sequence_length) means we are looking at the next (sequence_length) rows,
    but we often want just pairs: row i and row i+1. So you can adapt as needed.
    """
    # Example: for sequence_length=2, we'll look at pairs of consecutive actions
    pairs = list(zip(data['action'][:-1], data['action'][1:]))
    pair_strings = [f"{a1} -> {a2}" for (a1, a2) in pairs]
    sequence_counts = pd.Series(pair_strings).value_counts().head(15)  # Top 15
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sequence_counts.index, y=sequence_counts.values)
    plt.title(f"Most Frequent {sequence_length}-Action Sequences")
    plt.xlabel(f"Action Sequences ({sequence_length}-actions)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


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


def plot_action_vs_eyeblink_count(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="action", y="eyeblink_count", data=data)
    plt.title("Action vs Eyeblink Count")
    plt.xlabel("Action")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_action_heatmap(data):
    """
    Note: This assumes 'coords' column is in the format 'x, y'.
    Extract integer or float coordinates, then plot a 2D hexbin.
    """
    data[['x', 'y']] = data['coords'].str.extract(r'(\d+\.?\d*),\s*(\d+\.?\d*)').astype(float)
    plt.figure(figsize=(10, 6))
    plt.hexbin(data['x'], data['y'], gridsize=30, cmap='YlGnBu')
    plt.colorbar(label="Frequency")
    plt.title("Action Heatmap")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    plt.show()


def action_eyeblink_correlation(data):
    action_dummies = pd.get_dummies(data['action'])
    action_dummies['eyeblink_count'] = data['eyeblink_count']
    correlation = action_dummies.corr()['eyeblink_count'].sort_values(ascending=False)
    print("Correlation between each action dummy and Eyeblink Count:")
    print(correlation)


def plot_action_trends_over_time(data):
    """
    Plots how each action’s frequency changes by hour of the day.
    """
    data['hour'] = data['timestamp'].dt.hour
    action_trends = data.groupby(['hour', 'action']).size().unstack(fill_value=0)
    action_trends.plot(kind='line', stacked=True, figsize=(12, 8))
    plt.title("Action Frequency by Hour of the Day")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Action Frequency")
    plt.tight_layout()
    plt.show()


def action_timing_analysis(data):
    data['time_diff'] = data['timestamp'].diff().dt.total_seconds()
    action_timing = data.groupby('action')['time_diff'].mean()
    print("Average time difference between actions, by Action:")
    print(action_timing)


# ---------------------------------------
# 3. Additional Tracking
# ---------------------------------------
def plot_inactivity_periods(data, inactivity_threshold=5.0):
    """
    Identifies and plots periods of inactivity longer than `inactivity_threshold` seconds.
    """
    data['time_diff'] = data['timestamp'].diff().dt.total_seconds()
    # Mark those periods where time_diff > threshold
    inactivity_periods = data[data['time_diff'] > inactivity_threshold]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(inactivity_periods['timestamp'], inactivity_periods['time_diff'], color='red')
    plt.axhline(y=inactivity_threshold, color='gray', linestyle='--')
    plt.title(f"Inactivity Periods > {inactivity_threshold} seconds")
    plt.xlabel("Time")
    plt.ylabel("Inactivity Duration (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_rolling_eyeblink(data, window_size=10):
    """
    Plots the rolling average of the user's eyeblink count (over a given window_size).
    """
    data['rolling_blinks'] = data['eyeblink_count'].rolling(window_size).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["eyeblink_count"], alpha=0.3, label="Raw Eyeblink")
    plt.plot(data["timestamp"], data["rolling_blinks"], color='red', label="Rolling Average")
    plt.title(f"Rolling Average Eyeblink Count (Window = {window_size})")
    plt.xlabel("Time")
    plt.ylabel("Eyeblink Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def build_transition_matrix(data):
    """
    Constructs a transition matrix: the frequency of going from Action i to Action j.
    """
    actions = data['action']
    # shift(-1) means next row’s action
    next_actions = actions.shift(-1)

    # Create a DataFrame with current and next action
    transitions = pd.DataFrame({'current': actions, 'next': next_actions}).dropna()

    # Build a frequency table
    transition_counts = pd.crosstab(transitions['current'], transitions['next'])
    
    # Normalize to get probabilities (row-wise)
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    print("Transition Matrix (row-wise probabilities):")
    print(transition_matrix)
    return transition_matrix


# ---------------------------------------
# Main Code to Run
# ---------------------------------------
if __name__ == "__main__":
    log_file_path = '13-48_23-12_log.txt'
    data = load_log_data(log_file_path)

    # Original
    plot_eyeblink_count_over_time(data)
    plot_most_frequent_actions(data)
    plot_action_sequences(data, sequence_length=2)
    plot_action_duration(data)
    plot_action_vs_eyeblink_count(data)
    plot_action_heatmap(data)
    action_eyeblink_correlation(data)
    plot_action_trends_over_time(data)
    action_timing_analysis(data)

    # New additions
    plot_inactivity_periods(data, inactivity_threshold=5.0)
    plot_rolling_eyeblink(data, window_size=10)
    tm = build_transition_matrix(data)
