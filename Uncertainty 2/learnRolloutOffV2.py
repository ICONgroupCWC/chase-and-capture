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

from src.constants import SpiderAndFlyEnv, deter_5050Prey, AgentType, \
    QnetType
from src.qnetwork_coordinated import QNetworkCoordinated
from src.agent_seq_rollout import SeqRolloutAgent
from gym.envs.registration import register

import random
import wandb
import warnings
from pymongo import MongoClient
# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)

register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)


SEED = 42

M_AGENTS = 2
P_PREY = 2

N_SAMPLES = 50000
BATCH_SIZE = 1024
EPOCHS = 5000
N_SIMS_MC = 50
FROM_SCRATCH = True
INPUT_QNET_NAME = deter_5050Prey
OUTPUT_QNET_NAME = deter_5050Prey
BASIS_POLICY_AGENT = AgentType.RULE_BASED
QNET_TYPE = QnetType.BASELINE

def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 

def insertToMongo(dictInsert):
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    # Select the database and collection
    db = client['sequentialdata']  # Replace 'your_database_name' with your actual database name
    collection = db[f'mode_30_Full']  # Replace 'your_collection_name' with your actual collection name
    collection.insert_one(dictInsert)
    return



def retrieveRandomFromMongo(num_entries, query=None):
    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")  # Adjust the connection string if needed
    
    # Select the database and collection
    db = client['sequentialdata']  # Replace with your actual database name
    collection = db['mode_11']  # Replace with your actual collection name
    
    # If no query is provided, retrieve all documents
    if query is None:
        query = {}
    
    # Retrieve all documents matching the query
    all_documents = list(collection.find(query))
    # Randomly sample the specified number of entries
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
        action_q_values = np.array(doc["qval"], dtype=np.float32)
        outputSamples.append((x, action_q_values))

    return outputSamples


import random
import string

# Function to generate a random code word
def generate_code_word(length=8):
    letters = string.ascii_letters  # Includes both lowercase and uppercase letters
    return ''.join(random.choice(letters) for _ in range(length))



def generate_samples(n_samples):
    print('Started sample generation.')

    samples = []

    # create Spider-and-Fly game
    env = gym.make(SpiderAndFlyEnv)

    # TODO Switched off for re-training
    # env.seed(seed)

    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:
            # init env
            obs_n = env.reset()
            # visualize_image(env.render())

            # init agents
            m_agents = env.n_agents
            p_preys = env.n_preys
            grid_shape = env.grid_shape
            action_space = env.action_space[0]

            agents = [SeqRolloutAgent(
                i, m_agents, p_preys, grid_shape, env.action_space[i],
                n_sim_per_step=N_SIMS_MC,
                basis_agent_type=BASIS_POLICY_AGENT,
                qnet_type=QNET_TYPE,
            ) for i in range(m_agents)]

            # init stopping condition
            done_n = [False] * m_agents
            epi_steps = 0
            epi_code = generate_code_word()
            while not all(done_n):
                prev_actions = {}
                prev_action_qVals = {}
                act_n = []
                for i, (agent, obs) in enumerate(zip(agents, obs_n)):
                    best_action, action_q_values = agent.act_with_info(
                        obs, prev_actions=prev_actions)

                    # create an (x,y) sample for QNet
                    agent_ohe = np.zeros(shape=(m_agents,), dtype=np.float32)
                    agent_ohe[i] = 1.

                    prev_actions_ohe = np.zeros(shape=(m_agents * action_space.n,), dtype=np.float32)

                    for agent_i, action_i in prev_actions.items():
                        ohe_action_index = int(agent_i * action_space.n) + prev_actions[agent_i]
                        prev_actions_ohe[ohe_action_index] = 1.

                    obs_first = np.array(obs, dtype=np.float32).flatten()  # same for all agent
                    x = np.concatenate((obs_first, agent_ohe, prev_actions_ohe))



                    
                    prevActQs = []
                    for prevAgent in list(prev_action_qVals.keys()):
                        prevQVal = prev_action_qVals[prevAgent].tolist()
                        prevActQs = prevActQs + prevQVal

                    dictInsert = {
                        "obs" : obs_first.tolist(),
                        "qval" :action_q_values.tolist(),
                        "prev_qvals" : prevActQs,

                    }

                    dictInsert = {
                        "episode" : epi_code,
                        "step" : epi_steps,
                        "obs" : obs_first.tolist(),
                        "agent": i,
                        "qval" :action_q_values.tolist(),
                        "prev_qvals" : prevActQs,
                        "prev_actions" : prev_actions_ohe.tolist()
                        }
                    


                    insertToMongo(dictInsert)

                    samples.append((x, action_q_values))
                    pbar.update(1)
                    if len(samples) == N_SAMPLES:
                        env.close()
                        return samples

                    # best action taken for the agent i
                    prev_actions[i] = best_action
                    prev_action_qVals[i] = action_q_values
                    act_n.append(best_action)

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)
                epi_steps += 1
                # visualize_image(env.render())


    env.close()

    print('Finished sample generation.')

    return samples[:n_samples]



if __name__ == '__main__':
    t1 = perf_counter()
    n_workers = 10
    chunk = int(N_SAMPLES / n_workers)
    train_samples = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for _ in range(n_workers):
            futures.append(pool.submit(generate_samples, chunk))

        for f in as_completed(futures):
            samples_part = f.result()
            train_samples += samples_part
