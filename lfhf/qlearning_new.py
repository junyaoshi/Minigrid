from pprint import pprint
import heapq
from PIL import Image
from copy import deepcopy

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX

alpha = 1
gamma = 0.9

def find_goal(world_grid):
    width, height, _ = world_grid.shape
    for i in range(width):
        for j in range(height):
            if world_grid[i, j, 0] == OBJECT_TO_IDX["goal"]:
                return i, j
    return None

def find_agent(world_grid):
    width, height, _ = world_grid.shape
    for i in range(width):
        for j in range(height):
            if world_grid[i, j, 0] == OBJECT_TO_IDX["agent"]:
                return i, j, world_grid[i, j, 2]
    return None

def choose_action(q_table, world_grid):
    """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice."""
    q_values_of_state = q_table[find_agent(world_grid)]
    maxValue = max(q_values_of_state.values())
    action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

    return action

def play(env_name, q_table, width, height, num_dirs, available_actions):
    reward_table = deepcopy(q_table)
    new_q_table = deepcopy(q_table)
    for x in range(1, width-1):
        for y in range(1, height-1):
            for dir in range(num_dirs):
                for action in available_actions:
                    env: MiniGridEnv = gym.make(env_name, agent_start_pos=(x, y), agent_start_dir=dir, render_mode="rgb_array")
                    env = FullyObsWrapper(env)
                    obs, _ = env.reset()
                    world_grid = obs["image"]
                    old_state = find_agent(world_grid)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    world_grid = obs["image"]
                    new_state = find_agent(world_grid)
                    reward_table[(x, y, dir)][action] = reward

                    q_values_of_state = q_table[new_state]
                    max_q_value_in_new_state = max(q_values_of_state.values())
                    current_q_value = q_table[old_state][action]
                    new_q_table[old_state][action] = (1 - alpha) * current_q_value + alpha * (reward + gamma * max_q_value_in_new_state)

    return new_q_table

# def learn(q_table, old_state, reward, new_state, action):
#     """Updates the Q-value table using Q-learning"""
#     q_values_of_state = q_table[new_state]
#     max_q_value_in_new_state = max(q_values_of_state.values())
#     current_q_value = q_table[old_state][action]
    
#     q_table[old_state][action] = (1 - alpha) * current_q_value + alpha * (reward + gamma * max_q_value_in_new_state)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-Empty-Random-5x5-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    world_grid = obs["image"]
    width = env.width
    height = env.height
    num_dirs = 4
    available_actions = [env.actions.left, env.actions.right, env.actions.forward]

    q_table = dict()
    for x in range(width):
        for y in range(height):
            for dir in range(num_dirs):
                q_table[(x, y, dir)] = {env.actions.left:0, env.actions.right:0, env.actions.forward:0}
    

    q_diff = 10000
    q_threshold = 0.001
    step = 0
    while q_diff > q_threshold:
    # while step < 5:
        print('Step: ', step, ' q_diff: ', q_diff)
        q_diff = 0
        new_q_table = play(env_name, q_table, width, height, num_dirs, available_actions)
        for x in range(width):
            for y in range(height):
                for dir in range(num_dirs):
                    for action in available_actions:
                        q_diff = q_diff + abs(new_q_table[(x, y, dir)][action] - q_table[(x, y, dir)][action])
        q_table = new_q_table
        step += 1
    
    for x in range(1, width-1):
        for y in range(1, height-1):
            for dir in range(4):
                print(x, y, dir, q_table[(x, y, dir)])

    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        world_grid = obs["image"]
        state = find_agent(world_grid)
        print(q_table[state])
        img = env.render()
        img = Image.fromarray(img)
        img.save("obs.jpg")
        next = input('Press ENTER to go next: ')
        action = choose_action(q_table, world_grid)
        print(action)
        obs, reward, terminated, truncated, _ = env.step(action)
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    print('done')
