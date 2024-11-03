
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

INPUT_QNET_NAME = '/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/PPO/bert_marl/mode15-dqn/artifacts/action_pred/3_400_mode15_agent_1.pt'




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


    mode_name = 15
    IN = 10
    HL1 = 80
    HL2 = 180
    HL3 = 80
    OUT = 5

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
                    action_id = agent.act(obs, prev_actions=prev_actions)
                    

                    obs_first = np.array(obs, dtype=np.float32).flatten() 
                   
                    act_n.append(action_id)

                    x = torch.Tensor(np.array(obs, dtype=np.float32)).view(1,-1)
                    output = clf.predict(x).item()
                    prev_actions[i] = output


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

        env.close()
        results_experiment[experi] = steps_history

        with open("Approx_Seq_New_15.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)