from pprint import pprint
import heapq
from PIL import Image

import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX


def find_goal(world_grid):
    width, height, _ = world_grid.shape
    for i in range(width):
        for j in range(height):
            if world_grid[i, j, 0] == OBJECT_TO_IDX["goal"]:
                return i, j
    return None


def get_neighbors(world_grid, node):
    width, height, _ = world_grid.shape
    row, col = node
    neighbors = []
    if row > 0 and world_grid[row - 1, col, 0] != OBJECT_TO_IDX["wall"]:
        neighbors.append((row - 1, col))
    if row < width - 1 and world_grid[row + 1, col, 0] != OBJECT_TO_IDX["wall"]:
        neighbors.append((row + 1, col))
    if col > 0 and world_grid[row, col - 1, 0] != OBJECT_TO_IDX["wall"]:
        neighbors.append((row, col - 1))
    if col < height - 1 and world_grid[row, col + 1, 0] != OBJECT_TO_IDX["wall"]:
        neighbors.append((row, col + 1))
    return neighbors


def dijkstra(world_grid):
    visited = set()
    priority_queue = []

    width, height, _ = world_grid.shape
    dist_to_goal = np.ones((width, height)) * np.inf

    goal = find_goal(world_grid)
    assert goal is not None
    goal_row, goal_col = goal
    dist_to_goal[goal_row, goal_col] = 0
    heapq.heappush(priority_queue, (0, (goal_row, goal_col)))

    pbar = tqdm(desc="Dijkstra", total=width * height)
    while priority_queue:
        # go greedily by always extending the shorter cost nodes first
        _, node = heapq.heappop(priority_queue)
        visited.add(node)
        pbar.update(1)

        for neighbor in get_neighbors(world_grid, node):
            if neighbor in visited:
                continue

            row, col = node
            neighbor_row, neighbor_col = neighbor
            new_dist = dist_to_goal[row, col] + 1
            if new_dist < dist_to_goal[neighbor_row, neighbor_col]:
                dist_to_goal[neighbor_row, neighbor_col] = new_dist
                heapq.heappush(
                    priority_queue, (dist_to_goal[neighbor_row, neighbor_col], neighbor)
                )

    pbar.close()
    return dist_to_goal


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-FourRooms-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    world_grid = obs["image"]
    print("World grid:")
    pprint(world_grid[:, :, 0])

    # obs, *_ = env.step(env.actions.left)
    # world_grid = obs["image"]
    # pprint('World grid after moving left:')
    # pprint(world_grid[:, :, 0])

    # obs, *_ = env.step(env.actions.forward)
    # world_grid = obs["image"]
    # pprint('World grid after moving forward:')
    # pprint(world_grid[:, :, 0])

    dist_to_goal = dijkstra(world_grid)
    pprint("Distance to goal:")
    pprint(dist_to_goal)
