import argparse
from pprint import pprint

from PIL import Image
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

from feedback import noisy_sigmoid_feedback
from behavior import random_action, compute_coverage
from reward import dijkstra, compute_reward, get_all_reward_samples
from utils import print_world_grid, print_dist_to_goal, kl_divergence, str2bool
from model import train_model
from q_learning import q_learning_from_reward_model, q_learning_from_env_reward


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, default="results/test")

    # probe env
    parser.add_argument("--probe_env", type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--n_probe_samples", type=int, default=100)

    # feedback
    parser.add_argument("--noise", type=float, default=0.05)

    # model training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument(
        "--n_blocks", type=int, default=2, help="number of network blocks"
    )
    parser.add_argument("--residual", type=str2bool, default=True)

    # q-learning
    parser.add_argument(
        "--downstream_envs",
        nargs="+",
        default=[
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-FourRooms-v0",
        ],
    )
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    parser.add_argument(
        "--epsilon_linear_decay",
        type=str2bool,
        default=True,
        help="if true, epsilon decays linearly to epsilon_min, else stays constant",
    )
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()
    return args


def generate_samples(env: MiniGridEnv, args, world_grid, dist_to_goal):
    """Generate reward samples from the environment."""
    visited = set()
    reward_data = []
    for _ in range(args.n_probe_samples):
        action = random_action()
        state = (*env.agent_pos, env.agent_dir)
        visited.add((state, action))

        *_, terminated, truncated, _ = env.step(action)
        next_state = (*env.agent_pos, env.agent_dir)
        reward = compute_reward(dist_to_goal, state, next_state)
        reward_data.append(reward)

        if terminated or truncated:
            env.reset()

    reward_data = np.array(reward_data)
    coverage = compute_coverage(world_grid, visited)

    return reward_data, coverage


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.log_dir = f"results/probe={args.probe_env}_samples={args.n_probe_samples}"
    args.log_dir += f"_noise={args.noise}"
    writer = SummaryWriter(args.log_dir)

    # log args
    pprint(f"args: \n{args}")
    for arg in vars(args):
        writer.add_text(f"args/{str(arg)}", str(getattr(args, arg)))

    env: MiniGridEnv = gym.make(args.probe_env, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()

    # visualize environment
    writer.add_image("probe_env/initial_world", env.render(), dataformats="HWC")
    world_grid = obs["image"]
    print(f"Probe env: {args.probe_env}")
    print_world_grid(world_grid)

    # generate dense reward function
    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    # generate reward and feedback samples
    all_reward = get_all_reward_samples(args.probe_env, dist_to_goal)
    np.random.shuffle(all_reward)
    all_feedback = noisy_sigmoid_feedback(all_reward, args.noise)

    reward_data, coverage = generate_samples(env, args, world_grid, dist_to_goal)
    np.random.shuffle(reward_data)
    feedback_data = noisy_sigmoid_feedback(reward_data, args.noise)

    kl_div = kl_divergence(reward_data, all_reward)
    print(f"State x Action space coverage: {coverage:.4f}")
    print(f"KL Divergence between reward samples and ground truth: {kl_div:.4f}")

    # tensorboard logging
    writer.add_histogram("reward/all_reward", all_reward)
    writer.add_histogram("reward/reward_data", reward_data)
    writer.add_histogram("feedback/feedback_data", feedback_data)
    writer.add_histogram("feedback/all_feedback", all_feedback)
    writer.add_text("stats/coverage", str(coverage))
    writer.add_text("stats/reward_kl_div", str(kl_div))

    model = train_model(
        reward_data, feedback_data, all_reward, all_feedback, args, writer
    )

    for downstream_env in args.downstream_envs:
        seed = np.random.randint(0, 500)
        q_learning_from_reward_model(downstream_env, model, args, writer, seed)
        q_learning_from_env_reward(downstream_env, args, writer, seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    from time import sleep
    sleep(0.5)
