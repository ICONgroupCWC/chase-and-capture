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
N_SIMS = 200



class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)
    


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





Number_of_experiments = 1
episodes_per_experiments = 300

if __name__ == '__main__':

    results_experiment = {}
    for experi in range(Number_of_experiments):


        env = gym.make(SpiderAndFlyEnv)
        steps_history = []
        
        for epi in range (episodes_per_experiments):
            env.seed(epi)
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
                    prev_actions[i] = action_id

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

        with open("sequential_rollout_2.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)