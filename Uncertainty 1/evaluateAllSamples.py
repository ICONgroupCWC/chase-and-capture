from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import


import random
import wandb
import warnings
from pymongo import MongoClient

from src.qnetwork_coordinated import QNetworkCoordinated


def retrieveRandomFromMongo(num_entries, agentID,query=None):
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    
    # Select the database and collection
    db = client['sequentialdata']  # Replace with your actual database name
    collection = db[f'mode_11_Agent{agentID}']  # Replace with your actual collection name
    
    # If no query is provided, retrieve all documents
    if query is None:
        query = {}
    
    # Retrieve all documents matching the query
    all_documents = list(collection.find(query))
    # Randomly sample the specified number of entries
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
        outputSamples.append((x, np.array(doc["qval"], dtype=np.float32)))

    return outputSamples

    
def evaluateAll(netWorkName):

    trainingDataset = retrieveRandomFromMongo(50000, 1,query=None)

    net = QNetworkCoordinated(2, 2, 5)
    net.load_state_dict(torch.load(netWorkName))
    net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()
    criterion = nn.MSELoss()
    lossVal = []
    for trainData in trainingDataset:
        inputs = trainData[0]
        inputTensor = torch.Tensor(inputs).view((1,-1))
        labels = torch.Tensor(trainData[1]).view((1,-1))
        outputs = net(inputTensor)
        loss = criterion(outputs, labels)
        lossVal.append(loss.detach().item())

    return lossVal

if __name__ == '__main__':
    # epoch = 0   78.42591139516593
    # netWorkName = 'artifacts/models/epoch_' + str(epoch) + '_deter_5050_mode11_agent_1'
    # lossVals_0 = evaluateAll(netWorkName)

    # epoch = 9990  0.005740986446327946
    # netWorkName = 'artifacts/models/epoch_' + str(epoch) + '_deter_5050_mode11_agent_1'
    # lossVals_9990 = evaluateAll(netWorkName)
    # import ipdb; ipdb.set_trace()

    epoch = 5400
    netWorkName = 'artifacts/models/epoch_' + str(epoch) + '_deter_5050_mode11_agent_1'
    lossVals_5400 = evaluateAll(netWorkName)
    import ipdb; ipdb.set_trace()


