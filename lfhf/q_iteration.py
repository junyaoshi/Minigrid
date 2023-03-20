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

from utils import reset_env, print_world_grid
from utils import WALL, EAST, SOUTH, WEST, NORTH, GOAL, DIRS, N_DIRS, N_ACTIONS, LEFT, RIGHT, FORWARD

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

def play(env_name, q_table, width, height, alpha, gamma):
    reward_table = deepcopy(q_table)
    new_q_table = deepcopy(q_table)
    for x in range(1, width-1):
        for y in range(1, height-1):
            for dir in range(N_DIRS):
                for action in range(N_ACTIONS):
                    try:
                        env: MiniGridEnv = gym.make(env_name, agent_start_pos=(x, y), agent_start_dir=dir, render_mode="rgb_array")
                    except:
                        env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
                        env.agent_pos = (x, y)
                        env.agent_dir = dir
                    env = FullyObsWrapper(env)
                    obs, _ = env.reset()
                    # print(env.agent_pos, env.agent_dir)
                    world_grid = obs["image"]
                    # print_world_grid(world_grid)
                    if find_goal(world_grid) is None:
                        continue
                    old_state = find_agent(world_grid)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    world_grid = obs["image"]
                    new_state = find_agent(world_grid)
                    reward_table[x, y, dir, action] = reward
                    print(reward)

                    if old_state and new_state:
                        q_values_of_state = q_table[new_state[0], new_state[1], new_state[2], :]
                        max_q_value_in_new_state = max(q_values_of_state)
                        current_q_value = q_table[old_state[0], old_state[1], old_state[2], action]
                        new_q_table[old_state[0], old_state[1], old_state[2], action] = (1 - alpha) * current_q_value + alpha * (reward + gamma * max_q_value_in_new_state)

    return new_q_table

def update_Q(world_grid, q_table, reward_table, width, height, alpha, gamma):
    new_q_table = deepcopy(q_table)
    for x in range(1, width-1):
        for y in range(1, height-1):
            for dir in range(N_DIRS):
                for action in range(N_ACTIONS):
                    if world_grid[x, y, 0] == WALL or world_grid[x, y, 0] == GOAL:
                        continue
                    new_x, new_y, new_dir = step(world_grid, (x, y, dir), action)
                        
                    q_values_of_state = q_table[new_x, new_y, new_dir, :]
                    max_q_value_in_new_state = max(q_values_of_state)
                    current_q_value = q_table[x, y, dir, action]
                    new_q_table[x, y, dir, action] = (1 - alpha) * current_q_value + alpha * (reward_table[x, y, dir, action] + gamma * max_q_value_in_new_state)

    return new_q_table

def generate_q_table(env_name, world_grid, q_threshold, alpha, gamma):
    width, height, _ = world_grid.shape
    Q = np.zeros((width, height, N_DIRS, N_ACTIONS))
    R = generate_reward_table(world_grid)

    q_diff = np.inf
    step = 0
    while q_diff > q_threshold:
        new_Q = update_Q(world_grid, Q, R, width, height, alpha, gamma)
        q_diff = np.sum(np.absolute(new_Q - Q))
        Q = new_Q
        step += 1
        print('Step: ', step, ' q_diff: ', q_diff)

    return Q

def generate_reward_table(world_grid):
    width, height, _ = world_grid.shape
    R = np.zeros((width, height, N_DIRS, N_ACTIONS))
    goal_x, goal_y = find_goal(world_grid)
    assert goal_x != 0 and goal_y != 0 and goal_x != width - 1 and goal_y != height - 1
    if world_grid[goal_x-1, goal_y, 0] != WALL:
        R[goal_x-1, goal_y, EAST, FORWARD] = 1
    if world_grid[goal_x+1, goal_y, 0] != WALL:
        R[goal_x+1, goal_y, WEST, FORWARD] = 1
    if world_grid[goal_x, goal_y-1, 0] != WALL:
        R[goal_x, goal_y-1, SOUTH, FORWARD] = 1
    if world_grid[goal_x, goal_y+1, 0] != WALL:
        R[goal_x, goal_y+1, NORTH, FORWARD] = 1
    
    return R

def step(world_grid, state, action):
    width, height, _ = world_grid.shape
    x, y, dir = state
    new_x = x
    new_y = y
    new_dir = dir

    # Rotate left
    if action == LEFT:
        new_dir -= 1
        if new_dir < 0:
            new_dir += 4

    # Rotate right
    elif action == RIGHT:
        new_dir = (new_dir + 1) % 4

    # Move forward
    elif action == FORWARD:
        if dir == EAST and world_grid[x+1, y, 0] != WALL:
            new_x += 1
        if dir == SOUTH and world_grid[x, y+1, 0] != WALL:
            new_y += 1
        if dir == WEST and world_grid[x-1, y, 0] != WALL:
            new_x -= 1
        if dir == NORTH and world_grid[x, y-1, 0] != WALL:
            new_y -= 1

    return new_x, new_y, new_dir

if __name__ == "__main__":
    # np.set_printoptions(linewidth=200)

    # env_name = "MiniGrid-Empty-Random-5x5-v0"
    # env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    # env = FullyObsWrapper(env)
    # obs, _ = env.reset()
    # img = env.render()
    # img = Image.fromarray(img)
    # img.save("obs.jpg")

    # world_grid = obs["image"]
    # width = env.width
    # height = env.height
    # num_dirs = 4
    # available_actions = [env.actions.left, env.actions.right, env.actions.forward]

    # q_table = dict()
    # for x in range(width):
    #     for y in range(height):
    #         for dir in range(num_dirs):
    #             q_table[(x, y, dir)] = {env.actions.left:0, env.actions.right:0, env.actions.forward:0}
    

    # q_diff = 10000
    # q_threshold = 0.001
    # step = 0
    # while q_diff > q_threshold:
    # # while step < 5:
    #     print('Step: ', step, ' q_diff: ', q_diff)
    #     q_diff = 0
    #     new_q_table = play(env_name, q_table, width, height, num_dirs, available_actions)
    #     for x in range(width):
    #         for y in range(height):
    #             for dir in range(num_dirs):
    #                 for action in available_actions:
    #                     q_diff = q_diff + abs(new_q_table[(x, y, dir)][action] - q_table[(x, y, dir)][action])
    #     q_table = new_q_table
    #     step += 1
    
    # for x in range(1, width-1):
    #     for y in range(1, height-1):
    #         for dir in range(4):
    #             print(x, y, dir, q_table[(x, y, dir)])

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
    #     action = choose_action(q_table, world_grid)
    #     print(action)
    #     obs, reward, terminated, truncated, _ = env.step(action)
    # img = env.render()
    # img = Image.fromarray(img)
    # img.save("obs.jpg")

    print('done')
