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

import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX

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

if __name__ == "__main__":
    pass
