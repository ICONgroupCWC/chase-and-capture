import wandb
import matplotlib.pyplot as plt
import numpy as np


wandb.init(project="Search_and_Rescue",name="evolved11_on_mode11")

# Example: Number of experiments and episodes
num_experiments = 100
num_episodes = 30

for exp in range(num_experiments):
    # Generate random steps for each episode (replace this with real data)
    steps_per_episode = np.random.randint(50, 200, size=num_episodes)  # Replace with actual RL steps

    # Plot the scatter plot for this experiment
    plt.figure()
    plt.scatter(range(num_episodes), steps_per_episode, c='blue', label='Steps')
    plt.title(f"Experiment {exp+1}")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Solve")
    plt.legend()

    # Log the scatter plot to wandb
    wandb.log({f"experiment_{exp+1}": wandb.Image(plt)})

    # Clear the current plot to avoid overlapping in the next iteration
    plt.clf()

# Finish the run
wandb.finish()
