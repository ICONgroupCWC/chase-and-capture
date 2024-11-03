import json
from matplotlib import pyplot as plt


with open('result_autonomous_2Agent_iDQN.json', 'r') as file:
    data = json.load(file)



epochs = list(data.keys())
averages = [sum(values) / len(values) for values in data.values()]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, averages, marker='o')
plt.title('Average Values for Each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Value')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()