from time import perf_counter
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed
import matplotlib
import matplotlib.pyplot as plt
import json
from src.agent_rule_based import RuleBasedAgent
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

from src.constants import SpiderAndFlyEnv


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


from dqn2 import DQN as Agent





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

def get_action_and_info(agent, obs):
    return agent.act_with_info(obs)





if __name__ == "__main__":
    Number_of_experiments = 10
    episodes_per_experiments = 30
    env = gym.make(SpiderAndFlyEnv)

    results_experiment = {}

    for experi in range(Number_of_experiments):
        steps_history = []
        steps_num = 0
        for epi in range(episodes_per_experiments):
            startTime = time.time()
            frames = []
            epi_steps = 0
            obs_n = env.reset()
            logDict = {
                        "a_0" : "",
                        "aQ_0" : "",
                        "a_1" :"",
                        "aQ_1" :"",
                        "S"  : epi_steps,
                        "E"  : epi,
                    
                    }
            
            
            newFrame = add_title_to_frame(env.render(), logDict)
            frames.append(newFrame)

            # visualize_image(env.render())
            m_agents = env.n_agents
            p_preys = env.n_preys
            grid_shape = env.grid_shape



            
            agents = [Agent(8, env.action_space[0].n).to(torch.device(
                                                    "cuda" if torch.cuda.is_available() else
                                                    "cpu"
                                                        )) for i in range(m_agents)]
            Ruleagents = [RuleBasedAgent(i, m_agents, p_preys, grid_shape, env.action_space[i]) for i in range(m_agents)]

            done_n = [False] * m_agents
            total_reward = 0
            while not all(done_n):
                obs_first = np.array(obs_n[0], dtype=np.float32).flatten()
                act_n = []
                


                for i, (agent, obs) in enumerate(zip(agents, obs_n)):

                    if i%2 ==1:
                        agent.load_state_dict(torch.load('artifacts/basepolicy/intermediates.pth'))
                        agent.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                        agent.eval()

                        obs = obs[i*2:i*2 +2] + obs[m_agents*2:]  

                        action_id = agent(torch.tensor(obs, dtype=torch.float32, device=torch.device(
                                                                        "cuda" if torch.cuda.is_available() else
                                                                        "cpu"
                                                        )).unsqueeze(0)).max(-1).indices.view(-1)
                        

                        act_n.append(action_id.item())

                    else:
                        action_id, action_distances = Ruleagents[i].act_with_info(obs)
                        act_n.append(action_id)
            
                obs_n, reward_n, done_n, info = env.step(act_n)
                # add title to frame
                newFrame = add_title_to_frame(env.render(), logDict)
                frames.append(newFrame)

                epi_steps += 1
                steps_num += 1

                # visualize_image(env.render())
                total_reward += np.sum(reward_n)

            endTime = time.time()
            print(f"End of {epi}'th episode with {epi_steps} steps")
            # create_movie_clip(frames, f'RuleBased_Mode_11_Random_{epi}_V2.mp4', fps=10)
            resDict = {
                "StepsToSolve" : epi_steps,
                "compute_Time" : endTime-startTime,
            }

            steps_history.append(resDict)
            

            if (epi+1) % 10 ==0:
                print("Checpoint passed")
                # axes are (time, channel, height, width)
                create_movie_clip(frames, f"NN_A1__Rule_A0_A2{epi+1}.mp4", fps=10)
                

        env.close()

        results_experiment[experi] = steps_history

        with open("NN_A1__Rule_A0_A2.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)


        

            

        