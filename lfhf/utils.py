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
    env.place_agent()
    return obs


if __name__ == "__main__":
    pass
