from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from model import *
# from utils import *
from collections import deque

action_list = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                            np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 0.0]), np.array([-1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0]),
                            np.array([-1.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.2])]


def get_action(policy, state_list):
    with torch.no_grad():
        a = policy(state_list).max(1)[1].item()

    return action_list[a]


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    episode_reward = 0
    step = 0

    state = env.reset()
    state = rgb2gray(state)
    # state = state
    state_list = [state for _ in range(history_length)]
    count = 0
    while True:

        # TODO: preprocess the state in the same way than in in your preprocessing in train_agent.py
        state_list.pop(0)

        state_list.append(state)
        # print(state_list)
        state_list_list = np.array(state_list)
        # print(state_list_list)
        state_list_list = torch.tensor(state_list_list)

        state_list_list = state_list_list.reshape(-1, history_length, 96, 96)

        # TODO: get the action from your agent! If you use discretized actions you need to transform them to continuous
        # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])

        a = get_action(agent, state_list_list)

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        state = rgb2gray(state)
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    env.close()

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True

    n_test_episodes = 10  # number of episodes to test
    history_length = 4
    # TODO: load agent
    agent = Model(history_length)
    checkp = torch.load('policy1.pth', map_location='cpu')
    agent.load_state_dict(checkp)
    agent.eval()
    env = gym.make('CarRacing-v0').unwrapped
    for i in range(n_test_episodes):
        run_episode(env, agent, rendering=True, max_timesteps=1000)



    print('... finished')
