from pprint import pprint
import heapq
from PIL import Image

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.utils.window import Window

epsilon = 0.05
alpha = 0.1
gamma = 1

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

def choose_action(q_table, world_grid, available_actions, epsilon):
    """Returns the optimal action from Q-Value table. If multiple optimal actions, chooses random choice.
    Will make an exploratory random action dependent on epsilon."""
    if np.random.uniform(0,1) < epsilon:
        action = available_actions[np.random.randint(0, len(available_actions))]
    else:
        q_values_of_state = q_table[find_agent(world_grid)]
        maxValue = max(q_values_of_state.values())
        action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue])

    return action

def learn(q_table, old_state, reward, new_state, action):
    """Updates the Q-value table using Q-learning"""
    q_values_of_state = q_table[new_state]
    max_q_value_in_new_state = max(q_values_of_state.values())
    current_q_value = q_table[old_state][action]
    
    q_table[old_state][action] = (1 - alpha) * current_q_value + alpha * (reward + gamma * max_q_value_in_new_state)


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-Empty-Random-5x5-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    # img = env.render()
    # img = Image.fromarray(img)
    # img.save("obs.jpg")

    world_grid = obs["image"]
    print("World grid:")
    pprint(np.transpose(world_grid[:, :, 0]))
    pprint(np.transpose(world_grid[:, :, 1]))
    pprint(np.transpose(world_grid[:, :, 2]))

    q_table = dict()
    for x in range(env.width):
        for y in range(env.height):
            for dir in range(4):
                q_table[(x, y, dir)] = {env.actions.left:0, env.actions.right:0, env.actions.forward:0}
    available_actions = [env.actions.left, env.actions.right, env.actions.forward]
    
    
    num_trials = 1000
    reward_per_episode = []
    for trial in range(num_trials):
        terminated = False
        truncated = False
        print(find_agent(world_grid))
        cumulative_reward = 0
        step = 0
        while not (terminated or truncated):
            old_state = find_agent(world_grid)
            action = choose_action(q_table, world_grid, available_actions, epsilon)
            obs, reward, terminated, truncated, _ = env.step(action)
            world_grid = obs["image"]
            new_state = find_agent(world_grid)
            learn(q_table, old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1
        obs, _ = env.reset()
        world_grid = obs["image"]
        reward_per_episode.append(cumulative_reward)
    # print(reward_per_episode)
    # plt.plot(reward_per_episode)
    # plt.show()

    # obs, _ = env.reset()
    # terminated = False
    # truncated = False
    # while not (terminated or truncated):
    #     world_grid = obs["image"]
    #     state = find_agent(world_grid)
    #     print(q_table[state])
    #     img = env.render()
    #     img = Image.fromarray(img)
    #     img.save("obs.jpg")
    #     next = input('Press ENTER to go next: ')
    #     action = choose_action(q_table, world_grid, available_actions, 0)
    #     print(action)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    # img = env.render()
    # img = Image.fromarray(img)
    # img.save("obs.jpg")
    for x in range(1, env.width-1):
        for y in range(1, env.height-1):
            for dir in range(4):
                print(x, y, dir, q_table[(x, y, dir)])
    print('done')
