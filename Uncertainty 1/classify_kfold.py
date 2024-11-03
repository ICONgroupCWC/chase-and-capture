

from pymongo import MongoClient
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(IN, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

        self.linear2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)


        self.linear3 = nn.Linear(HL2, HL3)
        self.bn3 = nn.BatchNorm1d(HL3)

        self.linear4 = nn.Linear(HL3, OUT)
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
  

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network with specific WDECAY")
    parser.add_argument('--WDECAY', type=float, default=None, help='Weight decay value (default: 0.06)')
    parser.add_argument('--WBNAME', type=str, default=None, help='Name for the graph')
    return parser.parse_args()



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
        action = doc["action"]
        outputSamples.append((x, action))

    return outputSamples


def retrieve_ordered_documents(mode_name,episodeName):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_{mode_name}_Full']

    # Query to find documents where episode is 'ABBaKhHP' and agent is 1, sorted by 'step'
    query = {"episode": episodeName, "agent": 0}
    documents = collection.find(query).sort("step", 1)

    return list(documents)


def createSamples(mode_name,episodeList):
    outputSamples = []
    for episodeName in episodeList:
        listofDocs = retrieve_ordered_documents(mode_name,episodeName)
        for doc in listofDocs:
            x = np.array(doc["obs"], dtype=np.float32)
            action = doc["action"]

            outputSamples.append((x, action))

    return outputSamples


def retrieveEpisodeNames(mode_name, ):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_{mode_name}_Full']

    # Query to get all unique episode names
    episode_names = collection.distinct('episode')

    return episode_names


from collections import Counter
def calculate_class_weights(samples, num_classes):
    # Count the number of occurrences of each class
    class_counts = Counter([label for _, label in samples])
    total_samples = len(samples)

    # Inverse frequency: higher weight for underrepresented classes
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Convert the list to a tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights_tensor


def split_episodes(listofEpisodes, test_size=0.3, random_state=42):
    # Split episodes into training and testing datasets
    train_episodes, test_episodes = train_test_split(listofEpisodes, test_size=test_size, random_state=random_state, shuffle=True)
    
    return train_episodes, test_episodes

def k_fold_split(samples, k=5):
    # Initialize KFold object
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(samples):
        train_samples = [samples[i] for i in train_index]
        test_samples = [samples[i] for i in test_index]
        yield train_samples, test_samples



def runTestEvaluation(model, test_samples):
    predAccuList= []
    lossList = []

    samples = test_samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()  # Set the model to evaluation mode

    data_loader = torch.utils.data.DataLoader(samples, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # No need to track gradients during evaluation
        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds = model.predict(inputs)

            correct_pred_percentage = (preds == labels).sum().item()*100/BATCH_SIZE

            predAccuList.append(correct_pred_percentage)
            lossList.append(loss.item())

    return np.mean(predAccuList), np.mean(lossList)



if __name__ == "__main__":
# /Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/PPO/bert_marl/mode15-dqn/artifacts/action_pred/4_450_mode15_agent_1.pt

    args = parse_args()
    WDECAY = args.WDECAY
    WBNAME = args.WBNAME

    import wandb
    wandb.init(project="Train Mode 15",name=f"{WBNAME}_{WDECAY}",save_code=True)


    mode_name = 15
    IN = 10
    HL1 = 80
    HL2 = 180
    HL3 = 80
    OUT = 5

    BATCH_SIZE  = 5000
    DATA_SET = 25000
    EPOCHS = 500
    K_FOLDS = 5 



    queryStatement = {
        "agent" :0,
    }


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    samples = retrieveRandomFromMongo(DATA_SET,mode_name,queryStatement)
    print(len(samples))
    class_weights = calculate_class_weights(samples, OUT)
    print(class_weights)
    class_weights = class_weights.to(device) 



    for fold_idx, (train_samples, test_samples) in enumerate(k_fold_split(samples, k=K_FOLDS)):
        print(f"Starting fold {fold_idx + 1}/{K_FOLDS}")

        data_loader = torch.utils.data.DataLoader(train_samples,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True)

        clf = Network().to(device)

        optimizer = optim.Adam(clf.parameters(), lr=1e-2, weight_decay=WDECAY)



        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                            T_0 = 100,# Number of iterations for the first restart
                                            T_mult = 1, # A factor increases TiTi​ after a restart
                                            eta_min = 1e-8) # Minimum learning rate
    
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    

        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
    


        avgTestPredAccu = 0
        avgTestLoss = 0


        for epoch in range(EPOCHS):  # loop over the dataset multiple times

            total_norm = []
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
                        total_norm.append(param_norm.item() ** 2)   # Accumulate norm squared



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
                    'Norm before clipping' : np.mean(total_norm),
                    'Learning Rate': optimizer.param_groups[0]['lr'],
                    'avgTestLoss' : avgTestLoss,
                    'avgTestPredAccu' : avgTestPredAccu,
                    },step=epoch + (EPOCHS*fold_idx)) 
            
            print(f"Epoch {epoch + 1}/{EPOCHS}, Prediction Accuracy = {epochAvg_predPerc}%, Loss = {epochAvg_loss}")
        
                

            if epoch % 50 == 0:
                avgTestPredAccu, avgTestLoss = runTestEvaluation(clf, test_samples)
                print(f"Finished Test Evaluation {avgTestPredAccu}%  L_{avgTestLoss}")
                torch.save(clf.state_dict(), f'artifacts/action_pred/{fold_idx}_{epoch}_mode{mode_name}_agent_1.pt')


            scheduler.step()
    



    