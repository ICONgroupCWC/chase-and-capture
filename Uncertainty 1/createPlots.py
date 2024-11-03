
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the results_experiment JSON file
with open("results_experiment_rule_based.json", "r") as json_file:
    results_experiment = json.load(json_file)

# Convert the dictionary keys from string to integer for proper sorting
results_experiment = {int(k): v for k, v in results_experiment.items()}

# Number of experiments and episodes per experiment
num_experiments = len(results_experiment)
num_episodes = len(results_experiment[0])

# Prepare arrays for mean and variance calculations
means = []
variances = []

# Calculate mean and variance for each experiment
for experiment, steps in results_experiment.items():
    mean_steps = np.mean(steps)
    variance_steps = np.var(steps)
    means.append(mean_steps)
    variances.append(variance_steps)

# Convert lists to numpy arrays for further operations
means = np.array(means)
variances = np.array(variances)

# Calculate moving average and variance over 10 consecutive experiments
window_size = 3
moving_avg_means = np.convolve(means, np.ones(window_size)/window_size, mode='valid')
moving_avg_variances = np.convolve(variances, np.ones(window_size)/window_size, mode='valid')

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot moving average mean line
plt.plot(range(len(moving_avg_means)), moving_avg_means, color='red', label='Moving Average (Mean Steps)', linewidth=2)

# Shade the region between (mean - variance) and (mean + variance) for the moving average
plt.fill_between(range(len(moving_avg_means)), 
                 moving_avg_means - moving_avg_variances, 
                 moving_avg_means + moving_avg_variances, 
                 color='gray', alpha=0.3, label='Variance')

# Set y-axis limit to start from 0
plt.ylim(0, max(moving_avg_means + moving_avg_variances) + 10)

# Labels and Title
plt.title('Moving Average of Steps per Experiment with Variance')
plt.xlabel('Experiment (Windowed)')
plt.ylabel('Steps to Solve')
plt.legend()

# Show plot
plt.show()
