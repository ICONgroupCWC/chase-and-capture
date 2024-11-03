from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2
import matplotlib.pyplot as plt
import json
register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)

import gym
import numpy as np
from typing import List
import random

from src.constants import SpiderAndFlyEnv,deter_11_on_20 , AgentType, QnetType
from src.agent import Agent
from src.agent_random import RandomAgent
from src.agent_rule_based import RuleBasedAgent
from src.agent_seq_rollout import SeqRolloutAgent
from src.agent_qnet_based import QnetBasedAgent
from src.agent_std_rollout import StdRolloutMultiAgent
from src.iDQN import iQNetworkCoordinated

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
import torch


# get the latest available model file or manually select which file to load
import os
manual_number = 10
directory = "/Users/athmajanvivekananthan/WCE/JEPA - MARL/multi-agent/iDQN/bert_5050_mode20/bertsekas-marl/artifacts/models/MSE/"
files = os.listdir(directory)
filtered_files = [f for f in files if f.startswith("MSE_mode20_agent_") and f.endswith(".pt")]
file_numbers = [(f, int(f.split('_')[-1].replace('.pt', ''))) for f in filtered_files]
largest_file = max(file_numbers, key=lambda x: x[1])[0]
selected_file = f"MSE_mode20_agent_{manual_number}.pt"




AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 30
#GENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 50
SEED = 42
N_SIMS_MC = 50
INPUT_QNET_NAME = directory + largest_file



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
    net.load_state_dict(torch.load(netModelFile))
    net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    net.eval()

    obs_first = np.array(obs, dtype=np.float32).flatten()
    output = net(torch.tensor(obs_first).view((1,-1)))
    max_value = torch.max(output)
    max_indices = (output == max_value).nonzero(as_tuple=True)[0]
    return random.choice(max_indices.tolist())


def runTest(episodes_per_experiments):
    correctPredCounter = 0
    env = gym.make(SpiderAndFlyEnv)
    steps_history = []
    
    for epi in range (episodes_per_experiments):
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
                    prev_act = getPrevAction(obs,i,INPUT_QNET_NAME,env.n_agents,env.n_preys)
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
            # visualize_image(add_title_to_frame(env.render(), logDict))

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
            # create_movie_clip(frames, f"asyncAuto_2agent_11_{epi}.mp4", fps=10)

    env.close()
    results_experiment[experi] = steps_history

    return steps_history, correctPredCounter
    
Number_of_experiments = 100
episodes_per_experiments = 50

if __name__ == '__main__':

    results_experiment = {}
    for experi in range(Number_of_experiments):
        steps_history, correctPredCounter = runTest(episodes_per_experiments)
        results_experiment[experi] = {"stepsHistory" : steps_history, "correctPredCounter" : correctPredCounter }
        
        with open("cross_20_on_30_asyncAuto_2agent.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)
