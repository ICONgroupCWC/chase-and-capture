

from pymongo import MongoClient
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt



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
  
    def predict(self, x):
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)  # Convert to probabilities
        predicted_label = torch.argmax(probabilities, dim=1)  # Get the class with the highest probability
        
        return predicted_label
  


def retrieveRandomFromMongo(num_entries, mode_name,query=None):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_{mode_name}_Full']
    
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
    for doc in retDocs:
        x = np.array(doc["obs"], dtype=np.float32)
        prev_agents_action = np.argmax(doc["prev_actions"][0:5])
        outputSamples.append((x, prev_agents_action))

    return outputSamples


if __name__ == "__main__":

    import wandb
    wandb.init(project="Mode 30",name=f"classify_action",save_code=True)


    mode_name = 30
    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5

    BATCH_SIZE  = 20000
    DATA_SET = 50000
    EPOCHS = 10000


    queryStatement = {
        "agent" :1,
    }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    samples = retrieveRandomFromMongo(DATA_SET,mode_name,queryStatement)

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    
    clf = Network().to(device)
    optimizer = optim.Adam(clf.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                        T_0 = 1000,# Number of iterations for the first restart
                                        T_mult = 1, # A factor increases TiTi​ after a restart
                                        eta_min = 1e-8) # Minimum learning rate
    

    criterion = nn.CrossEntropyLoss()
    

    # Lists to store loss and accuracy over epochs
    losses = []
    accuracies = []

    # Initialize the plot

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        total_norm = 0
        running_loss = .0
        n_batches = 0
        pred_perc = []

        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = clf(inputs)

            
            loss = criterion(outputs, labels)

            loss.backward()

            for param in clf.parameters():
                if param.grad is not None:
                    # Compute the norm of the gradient for this parameter
                    param_norm = param.grad.data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2  # Accumulate norm squared



            optimizer.step()
            preds = clf.predict(inputs)
            correct_pred_percentage = (preds == labels).sum().item()*100/BATCH_SIZE
            pred_perc.append(correct_pred_percentage)

            

            running_loss += loss.item()
            n_batches += 1


        epochAvg_predPerc = np.mean(pred_perc)
        epochAvg_loss = running_loss / n_batches

        wandb.log({'loss':epochAvg_loss,
                   'Prediction Accuracy' : epochAvg_predPerc,
                   'Norm before clipping' : total_norm,
                   'Learning Rate': optimizer.param_groups[0]['lr'],
                   },step=epoch) 
        
        print(f"Epoch {epoch + 1}/{EPOCHS}, Prediction Accuracy = {epochAvg_predPerc}%, Loss = {epochAvg_loss}")
        
        if epoch % 10 ==0:
            torch.save(clf.state_dict(), f'artifacts/models/action_pred/{epoch}_mode{mode_name}_agent_1.pt')

        

        scheduler.step()
    



    

