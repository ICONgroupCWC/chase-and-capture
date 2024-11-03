from gym.envs.registration import register
import ma_gym.envs.predator_prey.predator_prey
import time
import cv2
import matplotlib.pyplot as plt
import json




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
warnings.filterwarnings("ignore", category=UserWarning)
import wandb
from pymongo import MongoClient

AGENT_TYPE = AgentType.SEQ_MA_ROLLOUT
N_EPISODES = 30
#GENT_TYPE = AgentType.SEQ_MA_ROLLOUT
QNET_TYPE = QnetType.REPEATED
BASIS_AGENT_TYPE = AgentType.RULE_BASED
N_SIMS = 50
SEED = 42
N_SIMS_MC = 50
GRID = 20



if GRID == 10:
    register(
        id='PredatorPrey10x10-v4',
        entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
        max_episode_steps=1000,
        reward_threshold=1.0,
    )
elif GRID == 20:
    register(
        id='PredatorPrey10x10-v4',
        entry_point='ma_gym.envs.predator_prey.predator_prey_2020:PredatorPrey',
        max_episode_steps=1000,
        reward_threshold=1.0,
    )



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


def insertToMongo(dictInsert,agentID):
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    # Select the database and collection
    db = client['sequentialdata']  # Replace 'your_database_name' with your actual database name
    collection = db[f'mode_15_Agent{agentID}']  # Replace 'your_collection_name' with your actual collection name
    collection.insert_one(dictInsert)
    return


def getOppositeAction(currentAction):
    if currentAction < 2:
        return currentAction +2 
    elif currentAction >=2 and currentAction <4 :
        return currentAction - 2 
    else:
        return currentAction
    

Number_of_experiments = 100
episodes_per_experiments = 50

if __name__ == '__main__':

    results_experiment = {}
    for experi in range(Number_of_experiments):


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
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    # each agent acts based on the same observation
                    action_id, action_distances = agent.act(obs, prev_actions=prev_actions)
                    

                    obs_first = np.array(obs, dtype=np.float32).flatten() 


                    prev_actions_ohe = np.zeros(shape=(env.n_agents * env.action_space[0].n,), dtype=np.float32)
                    for agent_i, action_i in prev_actions.items():
                        ohe_action_index = int(agent_i * env.action_space[0].n) + prev_actions[agent_i]
                        prev_actions_ohe[ohe_action_index] = 1.


                    dictInsert = {
                        "obs" : obs_first.tolist(),
                        "qval" :action_distances.tolist(),
                        "prev_actions" : prev_actions_ohe.tolist()
                    }


                    # insertToMongo(dictInsert,i)

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

                    prev_actions[i] = getOppositeAction(action_id)


                    act_n.append(action_id)

                    


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
                # create_movie_clip(frames, f"Seq_Mode_11_random_{epi+1}.mp4", fps=10)

        env.close()
        results_experiment[experi] = steps_history

        with open("sequential_adversary_op_2020.json", "w") as json_file:
            json.dump(results_experiment, json_file, indent=4)