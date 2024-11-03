from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2
import matplotlib.pyplot as plt
register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

import gym
import numpy as np
from typing import List


from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from src.agent import Agent
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_qnet_based import QnetBasedAgent
from src.agent_std_rollout import StdRolloutMultiAgent

import warnings

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 30
#GENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 10
SEED = 42
N_SIMS_MC = 50


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




if __name__ == '__main__':
    INPUT_QNET_NAME = '/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/PPO/bert_marl/mode30-dqn/bertsekas-marl/artifacts/models/action_pred/6600_mode30_agent_1.pt'

    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = Network().to(device)
    clf.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))
    clf.eval()



    env = gym.make(SpiderAndFlyEnv)
    
    for epi in range (N_EPISODES):
        frames = []
        startTime = time.time()
        obs_n = env.reset()
        visualize_image(env.render())
        agents = create_agents(env, AGENT_TYPE)
        
        done_n = [False] * env.n_agents
        total_reward = 0.
        epi_steps = 0
        acu_counter = 0
        while not all(done_n):
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                # each agent acts based on the same observation
                action_id = agent.act(obs, prev_actions=prev_actions)
                



                act_n.append(action_id)
                prev_actions[i] = action_id


            print(act_n)
            obs_n, reward_n, done_n, info = env.step(act_n)
            visualize_image(env.render())
            epi_steps += 1
            total_reward += np.sum(reward_n)
            frames.append(env.render())

        endTime = time.time()

        print(f'Episode {epi}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
        print(f"Accurate Predictions {acu_counter*100/epi_steps}")
        


        if (epi+1) % 10 ==0:
            print("Checpoint passed")
            # axes are (time, channel, height, width)
            # create_movie_clip(frames, f"ManhattanRuleBased_2_agents_{epi+1}.mp4", fps=10)

    env.close()