from pymongo import MongoClient
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd




class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layer_1)
    self.bn1 = nn.BatchNorm1d(hidden_layer_1)

    self.linear2 = nn.Linear(hidden_layer_1, hidden_layer_2)
    self.bn2 = nn.BatchNorm1d(hidden_layer_2)


    self.linear3 = nn.Linear(hidden_layer_2, hidden_layer_3)
    self.bn3 = nn.BatchNorm1d(hidden_layer_3)


    self.linear4 = nn.Linear(hidden_layer_3, output_dim)
    self.relu = nn.ReLU()


  def forward(self, x):
    x = self.linear1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.bn3(x)
    x = self.relu(x)
    x = self.linear4(x)
    return x
  

def test_model(clf, test_data_loader, criterion, device):
    clf.eval()  # Set the model to evaluation mode
    
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation for testing
        for data in test_data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Prediction accuracy
            correct_pred = (outputs.max(-1).indices == labels.max(-1).indices).sum().item()
            total_correct += correct_pred
            total_samples += len(labels)
    
    avg_loss = running_loss / len(test_data_loader)
    accuracy = total_correct * 100 / total_samples
    
    return avg_loss, accuracy


def retrieveRandomFromMongo(num_entries, agentID, lossFn, query=None):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_20_Q_Agent{agentID}'] 
    
    if query is None:
        query = {}

    all_documents = list(collection.find(query))
    if num_entries >= len(all_documents):
        # If num_entries is greater than or equal to the total number of documents, return all documents
        retDocs =  all_documents

    else:
        # Randomly select the required number of entries
        sampled_documents = random.sample(all_documents, num_entries)
        retDocs =  sampled_documents

    outputSamples = []
    if lossFn == "MSE":
        for doc in retDocs:
            x = np.array(doc["obs"], dtype=np.float32)
            prev_actions = np.array(doc["prev_qvals"], dtype=np.float32)
            outputSamples.append((x, prev_actions))

    return outputSamples



if __name__ == "__main__":
    INPUT_QNET_NAME = 'artifacts/3220_mode20_agent_1.pt'
    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = Network().to(device)
    clf.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))
    clf.eval()

    # Retrieve 50,000 random samples first
    total_samples = retrieveRandomFromMongo(50000, 1, "MSE")


    inputs = []
    labels = []
    for sample in total_samples:
        inputs.append(list(sample[0]))
        labels.append(list(sample[1]))

    inputs_T = torch.Tensor(inputs)
    labels_T = torch.Tensor(labels)
    outputs_T = clf(inputs_T)
    criterion = nn.MSELoss()
    loss = criterion(outputs_T, labels_T)













    # Initialize bins to store samples
    bins = [[] for _ in range(6)]

    # Bin boundaries
    bin_ranges = [(-4*(i+1), -4*i) for i in range(6)]

    # Binning the samples based on the prev_qvals values
    for sample in total_samples:
        prev_qvals = sample[1]  # This is the prev_qvals array
        min_of_prevQval = prev_qvals.min()
        for i, (lower_bound, upper_bound) in enumerate(bin_ranges):
            if min_of_prevQval >= lower_bound and min_of_prevQval < upper_bound:
                bins[i].append(sample)
                break  # Once a sample is assigned to a bin, stop checking further

    # Process each bin
    summary = []
    criterion = nn.MSELoss()

    for i, bin_samples in enumerate(bins):
        if not bin_samples:
            continue

        inputs = []
        labels = []
        for sample in bin_samples:
            inputs.append(list(sample[0]))
            labels.append(list(sample[1]))

        inputs_T = torch.Tensor(inputs)
        labels_T = torch.Tensor(labels)
        outputs_T = clf(inputs_T)
        
        # Calculate MSE Loss
        loss = criterion(outputs_T, labels_T)

        # Add summary for the bin
        summary.append({
            "bin": f'$gte: {bin_ranges[i][0]}, $lt: {bin_ranges[i][1]}',
            "sampleCount": len(bin_samples),
            "avgMSELoss": loss.item(),
        })

    df = pd.DataFrame(summary)
    total_sample_count = df['sampleCount'].sum()
    mean_mse_loss = df['avgMSELoss'].mean()

    # Create a summary DataFrame
    summary_row = pd.DataFrame({
        'bin': ['Total/Average'],
        'sampleCount': [total_sample_count],
        'avgMSELoss': [mean_mse_loss]
    })

    # Concatenate the summary row with the original DataFrame
    df = pd.concat([df, summary_row], ignore_index=True)
    print(df)

    import ipdb; ipdb.set_trace()






    

    
    


    
    

    
