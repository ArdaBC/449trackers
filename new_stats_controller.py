import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_controller_log(file_path):
    """
    Loads controller log data from a specified file path
    and returns it as a pandas DataFrame.
    """
    # The data format for each line:
    # [timestamp] - (l3_x, l3_y) - (r3_x, r3_y) - action - blink_count
    data = pd.read_csv(file_path, sep=" - ", header=None,
                       names=["timestamp", "l3_coords", "r3_coords", "action", "eyeblink_count"],
                       engine='python')

    # Remove square brackets around timestamp and convert to datetime
    data["timestamp"] = data["timestamp"].str.strip("[]")
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Extract float values for l3 and r3 coordinates
    data[['l3_x', 'l3_y']] = data['l3_coords'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)
    data[['r3_x', 'r3_y']] = data['r3_coords'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)

    # Ensure eyeblink_count is integer
    data["eyeblink_count"] = data["eyeblink_count"].astype(int)

    return data

def plot_controller_eyeblink_over_time(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data["timestamp"], data["eyeblink_count"], label="Eyeblink Count")
    plt.title("Controller Eyeblink Count Over Time")
    plt.xlabel("Time")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_controller_most_frequent_actions(data):
    action_counts = data['action'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values)
    plt.title("Most Frequent Controller Actions")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_controller_action_duration(data):
    data['time_diff'] = data['timestamp'].diff().dt.total_seconds()
    action_durations = data.groupby('action')['time_diff'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    action_durations.plot(kind='bar')
    plt.title("Average Action Duration (Controller)")
    plt.xlabel("Action")
    plt.ylabel("Average Duration (seconds)")
    plt.tight_layout()
    plt.show()

def plot_controller_action_vs_eyeblink_count(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="action", y="eyeblink_count", data=data)
    plt.title("Controller Action vs Eyeblink Count")
    plt.xlabel("Action")
    plt.ylabel("Eyeblink Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def controller_action_eyeblink_correlation(data):
    action_dummies = pd.get_dummies(data['action'])
    action_dummies['eyeblink_count'] = data['eyeblink_count']
    correlation = action_dummies.corr()['eyeblink_count'].sort_values(ascending=False)
    print("Correlation between each action dummy and Eyeblink Count (Controller):")
    print(correlation)


def plot_thumbstick_distributions(data):
    """
    Plot the distribution of the left and right thumbstick positions.
    Useful to see if the user tends to hold the stick in certain directions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left stick
    sns.scatterplot(x='l3_x', y='l3_y', data=data, ax=axes[0], alpha=0.5)
    axes[0].set_title("Left Thumbstick Distribution")
    axes[0].set_xlabel("L3 X")
    axes[0].set_ylabel("L3 Y")
    
    # Right stick
    sns.scatterplot(x='r3_x', y='r3_y', data=data, ax=axes[1], alpha=0.5, color='orange')
    axes[1].set_title("Right Thumbstick Distribution")
    axes[1].set_xlabel("R3 X")
    axes[1].set_ylabel("R3 Y")
    
    plt.tight_layout()
    plt.show()


def plot_thumbstick_heatmap(data):
    """
    2D histogram (or hexbin) of left stick or right stick usage.
    """
    plt.figure(figsize=(10, 6))
    plt.hexbin(data['l3_x'], data['l3_y'], gridsize=30, cmap='Blues')
    plt.colorbar(label="Frequency")
    plt.title("Left Thumbstick Heatmap")
    plt.xlabel("L3 X")
    plt.ylabel("L3 Y")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hexbin(data['r3_x'], data['r3_y'], gridsize=30, cmap='Reds')
    plt.colorbar(label="Frequency")
    plt.title("Right Thumbstick Heatmap")
    plt.xlabel("R3 X")
    plt.ylabel("R3 Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    controller_file_path = '14-15_23-12_controller_log.txt'
    controller_data = load_controller_log(controller_file_path)

    # Basic Plots
    plot_controller_eyeblink_over_time(controller_data)
    plot_controller_most_frequent_actions(controller_data)
    plot_controller_action_duration(controller_data)
    plot_controller_action_vs_eyeblink_count(controller_data)
    controller_action_eyeblink_correlation(controller_data)

    # Additional
    plot_thumbstick_distributions(controller_data)
    plot_thumbstick_heatmap(controller_data)
