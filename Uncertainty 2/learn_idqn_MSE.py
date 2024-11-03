from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import ma_gym  # register new envs on import

from src.constants import SpiderAndFlyEnv, deter_5050Prey_agent_1, AgentType, QnetType, \
l1_loss_agent_1, ce_loss_agent_1, mse_loss_agent_1


from src.iDQN import iQNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent
from gym.envs.registration import register

import random
import wandb
import warnings
from pymongo import MongoClient
# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)




import ma_gym.envs.predator_prey.predator_prey
import time
import cv2
import json
from typing import List
import random
from src.agent import Agent



register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)


SEED = 42

M_AGENTS = 2
P_PREY = 2
N_SAMPLES = 25000
BATCH_SIZE = 4096   
EPOCHS = 5000
N_SIMS_MC = 50
FROM_SCRATCH = False
BASIS_POLICY_AGENT = AgentType.RULE_BASED
QNET_TYPE = QnetType.BASELINE
AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 50

def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 


def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def add_title_to_frame(frame, logDict):
    
    img = Image.fromarray(frame)
    layer = Image.new('RGB', (max(img.size),int(max(img.size)*1.5)), (255,255,255))
    layer.paste(img)
    font = ImageFont.truetype("arial.ttf", 15)
    draw = ImageDraw.Draw(layer)

    draw.text((int(round(max(img.size)*0.05)), max(img.size)),f"E: {logDict['E']}  S: {logDict['S']}",(0,0,0),font=font)
    draw.text((int(round(max(img.size)*0.05)), int(max(img.size))+15),f"Agent_1 : {logDict['a_0']} __________ : {logDict['aQ_0']}",(0,0,0),font=font)
    draw.text((int(round(max(img.size)*0.05)), int(max(img.size))+30),f"Agent_2 : {logDict['a_1']} __________ : {logDict['aQ_1']}",(0,0,0),font=font)

    # Convert the new image back to a NumPy array
    frame_with_title = np.asarray(layer)
    
    return frame_with_title


def getPrevAction(obs,agentID,netModelFile,M_AGENTS,P_PREY):
    net = iQNetworkCoordinated(M_AGENTS, P_PREY, 5,agentID)
    if netModelFile is not None:
        net.load_state_dict(torch.load(netModelFile))
    net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    obs_first = np.array(obs, dtype=np.float32).flatten()
    output = net(torch.tensor(obs_first).view((1,-1)))
    max_value = torch.max(output)
    max_indices = (output == max_value).nonzero(as_tuple=True)[0]
    return random.choice(max_indices.tolist())



def runTest(episodes_per_experiments,trainingEpoch,inputQName):
    env = gym.make(SpiderAndFlyEnv)
    steps_history = []
    correctPredCounterHistory = []
    for epi in range (episodes_per_experiments):
        correctPredCounter = 0
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        epi_steps = 0
        logDict = {
                    "a_0" : "",
                    "aQ_0" : "",
                    "a_1" :"",
                    "aQ_1" :"",
                    "S"  : epi_steps,
                    "E"  : epi,
                
                }
        
        frames.append(add_title_to_frame(env.render(), logDict))
        # visualize_image(add_title_to_frame(env.render(), logDict))

        agents = create_agents(env, AGENT_TYPE)
        

        done_n = [False] * env.n_agents
        total_reward = 0.
        
        while not all(done_n):
            prev_actions = {}
            actual_prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                if i != 0:
                    prev_act = getPrevAction(obs,i,inputQName,env.n_agents,env.n_preys)
                    prev_actions[i-1] = prev_act

                    if prev_act == actual_prev_actions[i-1]:
                        correctPredCounter += 1


                action_id, action_distances = agent.act(obs, prev_actions=prev_actions)

                if i ==0 :
                    logDict["E"] = epi
                    logDict["S"] = epi_steps
                    logDict["a_0"] = action_id
                    logDict["aQ_0"] = action_distances


                if i ==1 :
                    logDict["E"] = epi
                    logDict["S"] = epi_steps
                    logDict["a_1"] = action_id
                    logDict["aQ_1"] = action_distances


                act_n.append(action_id)
                actual_prev_actions[i] = action_id

            obs_n, reward_n, done_n, info = env.step(act_n)
            epi_steps += 1
            total_reward += np.sum(reward_n)
            frames.append(add_title_to_frame(env.render(), logDict))

        endTime = time.time()
        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
        resDict = {
            "trainingEpoch" : trainingEpoch,
            "StepsToSolve" : epi_steps,
            "compute_Time" : endTime-startTime,
        }


        steps_history.append(resDict)
        correctPredCounterHistory.append(correctPredCounter)

        # create_movie_clip(frames, f"Seq_Mode_11_random_{epi+1}_V3.mp4", fps=10)
        
        if (epi+1) % 10 ==0:
            print("Checpoint passed")
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"Evaluation_asyncAuto_2agent_11_{epi}.mp4", fps=10)

    env.close()
    return steps_history, correctPredCounterHistory

def insertToMongo(dictInsert,agentID):
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    # Select the database and collection
    db = client['sequentialdata']  # Replace 'your_database_name' with your actual database name
    collection = db[f'mode_11_i_Agent_{agentID}']  # Replace 'your_collection_name' with your actual collection name
    collection.insert_one(dictInsert)
    return

def one_hot_max_with_tie_resolution(values):
    # Find the maximum value in the list
    max_value = max(values)
    
    # Get the indices of all occurrences of the max value (in case of a tie)
    max_indices = [i for i, v in enumerate(values) if v == max_value]
    
    # Resolve the tie by choosing a random index from the list of max_indices
    chosen_index = random.choice(max_indices)
    
    # Create a one-hot encoded list with 0s
    one_hot_encoded = [0] * len(values)
    
    # Set the chosen index to 1
    one_hot_encoded[chosen_index] = 1
    
    return one_hot_encoded



def retrieveRandomFromMongo(num_entries, agentID, lossFn, query=None):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_30_Q_Agent{agentID}'] 
    
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
    if lossFn == "CE" or lossFn == "L1":
        for doc in retDocs:
            x = np.array(doc["obs"], dtype=np.float32)
            prev_actions = np.array(one_hot_max_with_tie_resolution(np.array(doc["prev_qvals"], dtype=np.float32)), dtype=np.float32)
            outputSamples.append((x, prev_actions))

    elif lossFn == "MSE":
        for doc in retDocs:
            x = np.array(doc["obs"], dtype=np.float32)
            prev_actions = np.array(doc["prev_qvals"], dtype=np.float32)
            outputSamples.append((x, prev_actions))

    return outputSamples




def create_agents(
        env: gym.Env,
        agent_type: str,
) -> List[Agent]:
    # init env variables
    m_agents = env.n_agents
    p_preys = env.n_preys
    #grid_shape = env.grid_shape
    grid_shape = env.grid_shape

    return [SeqRolloutAgent(
        agent_i, m_agents, p_preys, grid_shape, env.action_space[agent_i],
        n_sim_per_step=N_SIMS, basis_agent_type=BASIS_AGENT_TYPE, qnet_type=QNET_TYPE,
    ) for agent_i in range(m_agents)]


def train_qnetwork(samples, criterion,qnetName):



    accurateActions = {}
    trainingEvaluations = {}
    print('Started Training.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Found device: {device}.')
    net = iQNetworkCoordinated(M_AGENTS, P_PREY, 5,1)

    # load initial position
    if not FROM_SCRATCH:
        net.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))
    
    net.to(device)

    net.train()  


    optimizer = optim.Adam(net.parameters(), lr=0.01)

    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = .0
        n_batches = 0

        for data in data_loader:
            # TODO optimize
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # logging
            running_loss += loss.item()
            n_batches += 1

        
        print(running_loss / n_batches)
        wandb.log({'loss':running_loss / n_batches},step=epoch) 

        if epoch == 0:
            steps_history, correctPredCounterHistory = runTest(10,epoch,None)
            trainingEvaluations[epoch] = steps_history
            accurateActions[epoch] = correctPredCounterHistory
            torch.save(net.state_dict(), f'artifacts/models/{lossFn}/{lossFn}_mode30_agent_1{STARTEPOCH+epoch}.pt')

        if epoch> 0 and epoch <= 100:
            torch.save(net.state_dict(), qnetName)
            torch.save(net.state_dict(), f'artifacts/models/{lossFn}/{lossFn}_mode30_agent_1{STARTEPOCH+epoch}.pt')
            steps_history, correctPredCounterHistory = runTest(10,epoch,qnetName)
            trainingEvaluations[epoch] = steps_history
            accurateActions[epoch] = correctPredCounterHistory
            with open(f"trainingEvaluations_2agent_{lossFn}.json", "w") as json_file:
                json.dump(trainingEvaluations, json_file, indent=4)

            with open(f"accuratePreds_2agent_{lossFn}.json", "w") as json_file:
                json.dump(accurateActions, json_file, indent=4)

        else:
            if epoch % 10 == 0:
                torch.save(net.state_dict(), qnetName)
                torch.save(net.state_dict(), f'artifacts/models/{lossFn}/{lossFn}_mode30_agent_1{STARTEPOCH+epoch}.pt')
                print(f"Starting Evaluations with {epoch}'th training epoch")
                steps_history, correctPredCounterHistory = runTest(10,epoch,qnetName)
                trainingEvaluations[epoch] = steps_history
                accurateActions[epoch] = correctPredCounterHistory
                
                with open(f"trainingEvaluations_2agent_{lossFn}.json", "w") as json_file:
                    json.dump(trainingEvaluations, json_file, indent=4)
                
                with open(f"accuratePreds_2agent_{lossFn}.json", "w") as json_file:
                    json.dump(accurateActions, json_file, indent=4)

    print('Finished Training.')

    return net, trainingEvaluations, accurateActions




if __name__ == '__main__':
    STARTEPOCH =  2700
   
    lossFn = "MSE"
    INPUT_QNET_NAME = f'artifacts/models/{lossFn}/{lossFn}_mode30_agent_1{STARTEPOCH}.pt'


    if lossFn == "CE":
        criterion = nn.CrossEntropyLoss()
        qnetName = ce_loss_agent_1

    elif lossFn == "L1":
        criterion = nn.L1Loss()
        qnetName = l1_loss_agent_1

    elif lossFn == "MSE":
        criterion = nn.MSELoss()
        qnetName = mse_loss_agent_1

    

    wandb.init(project="iDQN",name=f"mode_30_{lossFn}_loss")
    agent_samples = retrieveRandomFromMongo(50000, 1, lossFn, query=None)
    net, trainingEvaluations, accurateActions = train_qnetwork(agent_samples, criterion,qnetName)

    torch.save(net.state_dict(), qnetName)
    
    with open(f"trainingEvaluations_2agent_{lossFn}.json", "w") as json_file:
        json.dump(trainingEvaluations, json_file, indent=4)

    with open(f"accuratePreds_2agent_{lossFn}.json", "w") as json_file:
        json.dump(accurateActions, json_file, indent=4)

    
        
    wandb.finish()




