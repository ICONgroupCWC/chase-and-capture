
from pymongo import MongoClient
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd



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
        prevQVal = doc["prev_qvals"]
        outputSamples.append((x, prev_agents_action,prevQVal))

    return outputSamples




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
  



def getDatasetDistribution():
    dataset = retrieveRandomFromMongo(DATA_SET,mode_name,queryStatement)
    dictHist = {
        0 : 0,
        1 : 0,
        2 : 0,
        3 : 0,
        4 :0,
    }

    # {0: 11657, 1: 11553, 2: 10608, 3: 11312, 4: 4870}

    for data in dataset:
        dictHist[data[1]] += 1
    return dictHist


if __name__ == "__main__":
    mode_name = 30
    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5


    queryStatement = {
        "agent" :1,
    }

    DATA_SET = 50000
    INPUT_QNET_NAME = '/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/PPO/bert_marl/mode30-dqn/bertsekas-marl/artifacts/models/action_pred/6600_mode30_agent_1.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = Network().to(device)
    clf.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))

    clf.eval()

    dataset = retrieveRandomFromMongo(DATA_SET,mode_name,queryStatement)

    evaluationResults = []

    for data in dataset:
        correctPred = False
        input = torch.Tensor(data[0])
        output = clf.predict(input.view(1,-1)).item()
        label = data[1]
        prevQV = data[2]

        if output == label:
            correctPred = True

        dictT = {
            "label" : label,
            "pred" : output,
            "correctPred" : correctPred,
            "prevQ" : prevQV,

        }

        evaluationResults.append(dictT)

    dfeval = pd.DataFrame(evaluationResults)
    dfeval.to_csv("Evaluation_of_Dataset.csv")







    

    




