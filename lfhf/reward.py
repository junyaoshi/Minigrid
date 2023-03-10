from pprint import pprint
import heapq

import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

from utils import WALL, EAST, SOUTH, WEST, NORTH, GOAL, DIRS, N_DIRS, ACTIONS
from utils import print_dist_to_goal, print_world_grid


def find_goal_states(world_grid):
    """Find the goal states in the world grid."""
    width, height, _ = world_grid.shape
    goal_x, goal_y = None, None
    for i in range(width):
        for j in range(height):
            if world_grid[i, j, 0] == GOAL:
                goal_x, goal_y = i, j

    assert goal_x is not None and goal_y is not None
    goal_states = [(goal_x, goal_y, dir) for dir in DIRS]
    return goal_states


def get_neighbors(world_grid, state):
    width, height, _ = world_grid.shape
    x, y, dir = state
    neighbors = []

    # should not be at wall state (periphery of env)
    assert x != 0 and y != 0 and x != width - 1 and y != height - 1

    # turn left
    neighbors.append((x, y, dir - 1 if dir > 0 else 3))

    # turn right
    neighbors.append((x, y, dir + 1 if dir < 3 else 0))

    # move backward
    if dir == EAST and world_grid[x - 1, y, 0] != WALL:
        neighbors.append((x - 1, y, dir))
    elif dir == SOUTH and world_grid[x, y - 1, 0] != WALL:
        neighbors.append((x, y - 1, dir))
    elif dir == WEST and world_grid[x + 1, y, 0] != WALL:
        neighbors.append((x + 1, y, dir))
    elif dir == NORTH and world_grid[x, y + 1, 0] != WALL:
        neighbors.append((x, y + 1, dir))

    return neighbors


def dijkstra(world_grid):
    visited = set()
    priority_queue = []

    width, height, _ = world_grid.shape
    dist_to_goal = np.ones((width, height, N_DIRS)) * np.inf  # x, y, direction

    goal_states = find_goal_states(world_grid)
    for goal_state in goal_states:
        dist_to_goal[goal_state] = 0
        heapq.heappush(priority_queue, (0, goal_state))

    pbar = tqdm(desc="Dijkstra", total=width * height * N_DIRS)
    while priority_queue:
        # go greedily by always extending the shorter cost states first
        _, state = heapq.heappop(priority_queue)
        visited.add(state)
        pbar.update(1)

        for neighbor in get_neighbors(world_grid, state):
            if neighbor in visited:
                continue

            new_dist = dist_to_goal[state] + 1
            if new_dist < dist_to_goal[neighbor]:
                dist_to_goal[neighbor] = new_dist
                heapq.heappush(priority_queue, (dist_to_goal[neighbor], neighbor))

    pbar.close()
    return dist_to_goal


def compute_reward(dist_to_goal, state, next_state):
    """Get the reward for the given state->next_state transition."""
    return dist_to_goal[state] - dist_to_goal[next_state]


def get_all_reward_samples(env_name, dist_to_goal):
    """Get all the possible reward samples from the environment."""
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    world_grid = obs["image"]
    goal_states = find_goal_states(world_grid)

    print("Getting all reward samples.")
    reward_samples = []
    for x in tqdm(range(env.width), leave=True):
        for y in tqdm(range(env.height), leave=False):
            if world_grid[x, y, 0] == WALL:
                continue

            for dir in DIRS:
                state = (x, y, dir)
                if state in goal_states:
                    continue

                for action in ACTIONS:
                    temp_env: MiniGridEnv = gym.make(
                        env_name,
                        agent_start_pos=(x, y),
                        agent_start_dir=dir,
                        render_mode="rgb_array",
                    )
                    temp_env = FullyObsWrapper(temp_env)
                    temp_env.reset()
                    temp_env.step(action)

                    next_state = (*temp_env.agent_pos, temp_env.agent_dir)
                    reward = compute_reward(dist_to_goal, state, next_state)
                    reward_samples.append(reward)

    return np.array(reward_samples)


if __name__ == "__main__":
    from PIL import Image
    np.set_printoptions(linewidth=200)

    # env_name = "MiniGrid-FourRooms-v0"
    env_name = "MiniGrid-Empty-5x5-v0"
    # env_name = "MiniGrid-Empty-8x8-v0"
    # env_name = "MiniGrid-Empty-Random-5x5-v0"

    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    world_grid = obs["image"]
    print_world_grid(world_grid)

    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    print("East: ")
    print(np.transpose(dist_to_goal[:, :, EAST]))

    print("South: ")
    print(np.transpose(dist_to_goal[:, :, SOUTH]))

    print("West: ")
    print(np.transpose(dist_to_goal[:, :, WEST]))

    print("North: ")
    print(np.transpose(dist_to_goal[:, :, NORTH]))
