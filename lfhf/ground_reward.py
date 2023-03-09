import argparse

from PIL import Image
import numpy as np
import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

from feedback import noisy_sigmoid_feedback
from behavior import random_action, compute_coverage
from reward import dijkstra, compute_reward
from utils import print_world_grid, print_dist_to_goal


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--probe_env", type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--n_probe_samples", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.05)

    return parser.parse_args()


def generate_samples(env: MiniGridEnv, args):
    # dijkstra distance to goal
    obs, _ = env.reset()
    world_grid = obs["image"]
    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    # generate samples
    visited = set()
    reward_data = []
    for _ in range(args.n_probe_samples):
        action = random_action()
        state = (*env.agent_pos, env.agent_dir)

        visited.add((state, action))

        *_, terminated, truncated, _ = env.step(action)
        next_state = (*env.agent_pos, env.agent_dir)
        reward = compute_reward(dist_to_goal, state, next_state)
        if reward not in [-1, 0, 1]:
            print()
        reward_data.append(reward)

        if terminated or truncated:
            env.reset()

    reward_data = np.array(reward_data)
    print(reward_data)
    feedback_data = noisy_sigmoid_feedback(reward_data, args.noise)

    print(f"Coverage: {compute_coverage(world_grid, visited):.4f}")

    return reward_data, feedback_data


def main(args):
    env: MiniGridEnv = gym.make(args.probe_env, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()

    # visualize environment
    world_grid = obs["image"]
    print_world_grid(world_grid)
    Image.fromarray(env.render()).save("obs.jpg")

    reward_data, feedback_data = generate_samples(env, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
