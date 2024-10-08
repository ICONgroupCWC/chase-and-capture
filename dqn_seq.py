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


import ma_gym.envs.predator_prey.predator_prey
from typing import List
from src.agent import Agent
from src.agent_seq_rollout import SeqRolloutAgent
from src.constants import SpiderAndFlyEnv, AgentType, QnetType
from PIL import Image, ImageDraw, ImageFont

import cv2
import wandb
import time
from gym.envs.registration import register
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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





Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

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
    



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(-1).indices.view(-1)
    else:
        return torch.tensor([[env.action_space[0].sample()]], device=device, dtype=torch.long).view(-1)






def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss.item()


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 



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
    env = gym.make(SpiderAndFlyEnv)

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # import wandb

    # Initialize WandB (ensure you have logged in)
    # wandb.init(project="DQN Spider and Fly", name="DQN Spider and Fly Sequential")


    episode_durations = []
    losList = []
    BATCH_SIZE = 2000
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    # Get number of actions from gym action space
    n_actions = env.action_space[0].n
    # Get the number of state observations  
    state_n = env.reset()
    state = state_n[0]
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


    steps_done = 0
    best_loss = float('inf')

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 10000
    else:
        num_episodes = 50

    for i_episode in range(num_episodes):
        agents = create_agents(env, AGENT_TYPE)
        # Initialize the environment and get its state
        state_n = env.reset()
        visualize_image(env.render())
        state = state_n[0]

        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_loss = 0
        for t in count():
            print(t)
            prev_actions = {}
            act_n = []
            for i, (agent, obs) in enumerate(zip(agents, state_n)):
                
                if i == 0:
                    action_id, action_distances = agent.act(obs, prev_actions=prev_actions)
                    # need to alternate prev_action[i] here 
                    # between real action and model output
                    action = select_action(state)
                    # prev_actions[i]  = action.detach().item()
                    prev_actions[i]  = action_id


                    act_n.append(action_id)

                if i ==1 :
                    action_id, action_distances = agent.act(obs, prev_actions=prev_actions)

                    prev_actions[i] = action_id
                    act_n.append(action_id)
            print(act_n)
            observation_n, reward_n, done_n, info = env.step(act_n)
            visualize_image(env.render())
            
            observation = observation_n[0]
            reward = np.sum(reward_n)
            done = all(done_n)


            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # # Perform one step of the optimization (on the policy network)
            # lossVal = optimize_model()
            # if lossVal is not None:
            #     losList.append(lossVal)
            #     episode_loss += lossVal

            # # Soft update of the target network's weights
            # # θ′ ← τ θ + (1 −τ )θ′
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
            #     plot_durations()

            #     if (i_episode + 1) % 10 == 0:
            #         avg_episode_loss = episode_loss / (t + 1)  # Average loss over episode
            #         print(f"Episode {i_episode + 1}: Avg Loss = {avg_episode_loss}")

            #         # # Log to WandB
            #         # wandb.log({
            #         #     "episode": i_episode + 1,
            #         #     "avg_loss": avg_episode_loss,
            #         #     "episode_duration": t + 1,
            #         #     "eps_threshold": EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY),
            #         # })
            #         model_save_path = f"artifacts/sequential/intermediates.pth"
            #         torch.save(policy_net.state_dict(), model_save_path)
                    

            #     # Save model every 10th episode if loss is lower than the best_loss
                    
                    
            #         if avg_episode_loss < best_loss:
            #             best_loss = avg_episode_loss
            #             model_save_path = f"artifacts/sequential/best_model_episode_{i_episode + 1}.pth"
            #             torch.save(policy_net.state_dict(), model_save_path)
            #             print(f"Model saved with loss {best_loss} at episode {i_episode + 1}")


                break


    plt.ioff()
    plt.show()
    model_save_path = f"artifacts/sequential/final.pth"
    torch.save(policy_net.state_dict(), model_save_path)

