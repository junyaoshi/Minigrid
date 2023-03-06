from pprint import pprint
import heapq
from PIL import Image

import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX


"""
> The integer values to which the agent's direction is encoded are:
> 
> - `0`: East
> - `1`: South
> - `2`: West
> - `3`: North

> World grid encoding:
world_grid[x_pos, y_pos] = [
    OBJECT_TO_IDX[object_type], 
    COLOR_TO_IDX[object_color], 
    agent direction / objectstate
]
"""
N_DIRS = 4
EAST, SOUTH, WEST, NORTH = range(N_DIRS)
WALL = OBJECT_TO_IDX["wall"]


def find_goal_states(world_grid):
    """Find the goal states in the world grid."""
    width, height, _ = world_grid.shape
    goal_x, goal_y = None, None
    for i in range(width):
        for j in range(height):
            if world_grid[i, j, 0] == OBJECT_TO_IDX["goal"]:
                goal_x, goal_y = i, j

    assert goal_x is not None and goal_y is not None
    goal_states = [(goal_x, goal_y, dir) for dir in range(N_DIRS)]
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

    # move forward
    if dir == EAST and world_grid[x + 1, y, 0] != WALL:
        neighbors.append((x + 1, y, dir))
    elif dir == SOUTH and world_grid[x, y + 1, 0] != WALL:
        neighbors.append((x, y + 1, dir))
    elif dir == WEST and world_grid[x - 1, y, 0] != WALL:
        neighbors.append((x - 1, y, dir))
    elif dir == NORTH and world_grid[x, y - 1, 0] != WALL:
        neighbors.append((x, y - 1, dir))

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


def print_world_grid(world_grid):
    """Prints the world grid in a readable format.
    Using transpose because np array's indexing is [row, col] and we want [x, y].
    """
    print("World grid:")
    pprint(np.transpose(world_grid[:, :, 0]))


def print_dist_to_goal(dist_to_goal):
    """Prints the distance to goal in a readable format.
    Using transpose because np array's indexing is [row, col] and we want [x, y].
    """
    print("Distance to goal:")
    pprint(np.transpose(np.mean(dist_to_goal, axis=2).round(1)))


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-FourRooms-v0"
    # env_name = "MiniGrid-Empty-5x5-v0"
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

    # obs, *_ = env.step(env.actions.left)
    # world_grid = obs["image"]
    # pprint('World grid after moving left:')
    # pprint(world_grid[:, :, 0])

    # obs, *_ = env.step(env.actions.forward)
    # world_grid = obs["image"]
    # pprint('World grid after moving forward:')
    # pprint(world_grid[:, :, 0])

    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)
