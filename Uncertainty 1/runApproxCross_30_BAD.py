
import ma_gym.envs.predator_prey.predator_prey

from typing import List

from src.constants import SpiderAndFlyEnv
from src.agent import Agent
from src.agent_seq_rollout import SeqRolloutAgent



from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import matplotlib
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
import numpy as np
import gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ma_gym  

from src.constants import SpiderAndFlyEnv, AgentType, QnetType


import cv2
import wandb
import time
from gym.envs.registration import register
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image, ImageDraw, ImageFont
import numpy as np


register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 50
INPUT_QNET_NAME = '/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/PPO/bert_marl/mode30-dqn/bertsekas-marl/artifacts/models/3410_mode30_agent_1.pt'





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

if __name__ == "__main__":

    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = Network().to(device)
    clf.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))
    clf.eval()

    Number_of_experiments = 100
    episodes_per_experiments = 50


    results_experiment = {}
    for experi in range(Number_of_experiments):


        env = gym.make(SpiderAndFlyEnv)
        steps_history = []
        
        for epi in range (episodes_per_experiments):
            frames = []
            startTime = time.time()
            obs_n = env.reset()
            epi_steps = 0
            frames.append(env.render())
            # visualize_image(env.render())

            agents = create_agents(env, AGENT_TYPE)
            
            done_n = [False] * env.n_agents
            total_reward = 0.
            
            while not all(done_n):
                prev_actions = {}
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    action_id, action_distances = agent.act(obs, prev_actions=prev_actions)
                    

                    obs_first = np.array(obs, dtype=np.float32).flatten() 
                   
                    act_n.append(action_id)


                    x = torch.Tensor(np.array(obs, dtype=np.float32)).view(1,-1)
                    output = clf(x)
                    prev_actions[i] = output.max(1).indices.item()

                    # prev_actions[i] = np.random.randint(0,5)
                    # prev_actions[i] = action_id

                obs_n, reward_n, done_n, info = env.step(act_n)
                epi_steps += 1
                total_reward += np.sum(reward_n)
                frames.append(env.render())
                # visualize_image(env.render())

            endTime = time.time()

            print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
            resDict = {
                "StepsToSolve" : epi_steps,
                "compute_Time" : endTime-startTime,
            }

            steps_history.append(resDict)
            # create_movie_clip(frames, f"Seq_Mode_11_random_{epi+1}_V3.mp4", fps=10)
            
            if (epi+1) % 10 ==0:
                print("Checpoint passed")
                # axes are (time, channel, height, width)
                # create_movie_clip(frames, f"sequential_rollout_RANDOM_SIG{experi}_{epi+1}.mp4", fps=10)

        env.close()
        results_experiment[experi] = steps_history

        with open("approx_cross_rollout_3410_mode30.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)
