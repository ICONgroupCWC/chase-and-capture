import time
from typing import List
import cv2
import warnings
import logging

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gym
import ma_gym  # register new envs on import
from ma_gym.wrappers.monitor import Monitor

from src.constants import SpiderAndFlyEnv, AgentType
from src.agent_std_rollout import StdRolloutMultiAgent

N_EPISODES = 30
N_SIMS_PER_MC = 50

from gym.envs.registration import register
import warnings
import json

# Suppress the specific gym warning
warnings.filterwarnings("ignore", category=UserWarning)
import wandb


register(
    id='PredatorPrey10x10-v4',
    entry_point='ma_gym.envs.predator_prey.predator_prey:PredatorPrey',
    max_episode_steps=1000,
    reward_threshold=1.0,
)
gym.logger.set_level(logging.ERROR)  # Set gym logger level to ERROR

warnings.filterwarnings("ignore", category=UserWarning, module="gym")  # Ignore UserWarnings from gym


def create_movie_clip(frames: list, output_file: str, fps: int = 10):
    # Assuming all frames have the same shape
    height, width, layers = frames[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()

def visualize_image(img: np.ndarray, pause_time: float = 0.5):

    if not isinstance(img, np.ndarray):
        raise ValueError("The provided image is not a valid NumPy array")

    plt.imshow(img)
    plt.axis('off') 
    plt.show(block=False) 
    plt.pause(pause_time)  
    plt.close() 

Number_of_experiments = 1
episodes_per_experiments =50

if __name__ == '__main__':
    results_experiment = {}
    for experi in range(Number_of_experiments):


        # create Spider-and-Fly game
        env = gym.make(SpiderAndFlyEnv)
        
        steps_history = []
        # env = Monitor(env, directory='../artifacts/recordings', force=True, )
        steps_num = 0
        for i_episode in tqdm(range(episodes_per_experiments)):
            startTime = time.time()
            frames = []
            epi_steps = 0
            # init env
            # obs_n = env.reset()
            obs_n = env.reset()


            # init env variables
            m_agents = env.env.n_agents
            p_preys = env.env.n_preys
            grid_shape = env.env._grid_shape

            # init agents
            std_rollout_multiagent = StdRolloutMultiAgent(
                m_agents, p_preys, grid_shape, env.action_space[0], N_SIMS_PER_MC)

            # init stopping condition
            done_n = [False] * env.n_agents

            total_reward = .0

            # run an episode until all prey is caught
            while not all(done_n):
                act_n = std_rollout_multiagent.act_n(obs_n)

                # update step
                obs_n, reward_n, done_n, info = env.step(act_n)
                epi_steps += 1
                steps_num += 1

                total_reward += np.sum(reward_n)
                # visualize_image(imgs)
                frames.append(env.render())

                # time.sleep(0.5)

            endTime = time.time()
            print(f'Episode {i_episode}: Reward is {total_reward}, with steps {epi_steps} exeTime{endTime-startTime}')
            resDict = {
                    "StepsToSolve" : epi_steps,
                    "compute_Time" : endTime-startTime,
                }
            steps_history.append(resDict)
            # create_movie_clip(frames, 'standardMARollout.mp4', fps=10)

            if (i_episode+1) % 10 ==0:
                print("Checpoint passed")
                # axes are (time, channel, height, width)
                create_movie_clip(frames, f"standardRollout_2_agents_{i_episode+1}.mp4", fps=10)

        # time.sleep(2.)

        env.close()
        results_experiment[experi] = steps_history

    with open("results_experiment_standardRollout.json", "w") as json_file:
        json.dump(results_experiment, json_file, indent=4)
