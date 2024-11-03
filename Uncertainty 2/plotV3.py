import json
from matplotlib import pyplot as plt
import pandas as pd

# Function to load and process data
def load_and_process_data(filename, window_size=200):
    with open(filename, 'r') as file:
        data = json.load(file)

    # Extract the data from the JSON file
    listOfWholeTraining = []
    epochs = list(data.keys())
    i = 0
    for trainEpoch in epochs:
        epochData = data[str(trainEpoch)]
        for indData in epochData:
            steps = indData['StepsToSolve']
            listOfWholeTraining.append({"episode": i, "steps": steps})
            i += 1

    dfTrainData = pd.DataFrame(listOfWholeTraining)

    # Group by 'episode' and calculate the mean 'steps' for each episode
    df_mean = dfTrainData.groupby('episode')['steps'].mean().reset_index()

    # Apply a rolling window to smooth the data
    df_mean['smoothed_steps'] = df_mean['steps'].rolling(window=window_size, min_periods=1).mean()

    return df_mean

# Load and process both datasets
df_async = load_and_process_data('cross_11_on_20_asyncAuto_2agent.json')
df_sequential = load_and_process_data('results_experiment_sequential_rollout.json')

# Plot both curves in the same figure
plt.figure(figsize=(10, 6))

# Plot asyncAuto data
plt.plot(df_async['episode'], df_async['smoothed_steps'], label='Async Auto (2 Agents)', color='blue', marker='o', linestyle='-')

# Plot sequential rollout data
plt.plot(df_sequential['episode'], df_sequential['smoothed_steps'], label='Sequential Rollout', color='red', marker='o', linestyle='-')

# Set labels and title
plt.xlabel('Episode')
plt.ylabel('Smoothed Average Steps')
plt.title('Comparison of Async Auto (2 Agents) vs Sequential Rollout')

# Show grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
