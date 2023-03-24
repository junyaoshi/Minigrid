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

> Action encoding:
- `0`: turn left
- `1`: turn right
- `2`: move forward
]
"""
from pprint import pprint
import argparse
import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.special import rel_entr

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.minigrid_env import MiniGridEnv


N_DIRS = 4
EAST, SOUTH, WEST, NORTH = range(N_DIRS)
DIRS = [EAST, SOUTH, WEST, NORTH]

WALL = OBJECT_TO_IDX["wall"]
GOAL = OBJECT_TO_IDX["goal"]

LEFT, RIGHT, FORWARD = 0, 1, 2
ACTIONS = [LEFT, RIGHT, FORWARD]
N_ACTIONS = len(ACTIONS)


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


def kl_divergence(samples, ground_truth):
    """Computes the KL divergence between two sets of samples.
    Args:
        samples (np.ndarray): random samples.
        ground_truth (np.ndarray): all ground truth samples.
    Returns:
        float: KL divergence between the two distributions.
    """
    gt_classes, gt_counts = np.unique(ground_truth, return_counts=True)
    gt_probs = gt_counts / len(ground_truth)
    sample_probs = np.array(
        [np.count_nonzero(samples == c) / len(samples) for c in gt_classes]
    )
    return sum(rel_entr(sample_probs, gt_probs))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_agent_state(env: MiniGridEnv):
    """Get the agent's state in the world grid."""
    return (*env.agent_pos, env.agent_dir)


def reset_env(env: MiniGridEnv, seed):
    """Reset the environment and return the initial state."""
    obs, _ = env.reset(seed=seed)
    randint = np.random.randint(0, 500)
    for _ in range(randint):
        env.place_agent()
    return obs


def hwc2chw(img):
    """Converts an image from HWC to CHW format."""
    return np.transpose(img, axes=(2, 0, 1))


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


def compute_coverage(world_grid, visited):
    """Computes coverage of the visited state, action pairs
    over all state action pairs."""
    assert isinstance(visited, set)
    total = (
        len(
            [
                cell[0]
                for row in world_grid
                for cell in row
                if cell[0] != WALL and cell[0] != GOAL
            ]
        )
        * N_DIRS
        * N_ACTIONS
    )
    assert len(visited) <= total, "more visited states than possible states"
    return len(visited) / total


class AverageMeter(object):
    """Computes and stores running average of a scalar value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ArrayAverageMeter(object):
    """Computes and stores running average of an array of values."""
    def __init__(self, shape):
        self.reset(shape)

    def reset(self, shape):
        self.avg = np.zeros(shape)
        self.sum = np.zeros(shape)
        self.cnt = np.zeros(shape)

    def update(self, inds, val, n=1):
        self.sum[inds] += val * n
        self.cnt[inds] += n
        self.avg[inds] = self.sum[inds] / self.cnt[inds]


if __name__ == "__main__":
    pass
