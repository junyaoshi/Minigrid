import argparse
from pprint import pprint

from PIL import Image
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

from feedback import generate_feedback
from behavior import random_action
from utils import kl_divergence, str2bool, compute_coverage, reset_env
from model import train_model
from q_learning import (
    visualize_policy,
    visualize_q_model_policy,
    q_value_iteration,
    q_learning_from_env_reward,
    q_learning_from_q_model,
    q_learning_from_true_q,
    q_learning_from_feedback_as_q,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # run config
    parser.add_argument("--log_dir", type=str, default="results_q")
    parser.add_argument("--debug", type=str2bool, default=True)

    # probe env
    parser.add_argument("--probe_env", type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--probe_q_threshold", type=float, default=0.001)
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
            # "MiniGrid-Empty-8x8-v0",
            # "MiniGrid-FourRooms-v0",
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
    parser.add_argument("--hd_feedback", type=str, default="base2")

    args = parser.parse_args()
    return args


def generate_samples(env: MiniGridEnv, args, world_grid, q_table):
    """Generate reward samples from the environment."""
    visited = set()
    q_data = []
    for _ in range(args.n_probe_samples):
        action = random_action()
        state = (*env.agent_pos, env.agent_dir)
        visited.add((state, action))

        *_, terminated, truncated, _ = env.step(action)
        q = q_table[state[0], state[1], state[2], action]
        q_data.append(q)

        if terminated or truncated:
            env.reset()

    q_data = np.array(q_data)
    coverage = compute_coverage(world_grid, visited)

    return q_data, coverage


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.debug:
        args.log_dir = f"{args.log_dir}/debug"
    else:
        args.log_dir = (
            f"{args.log_dir}/probe={args.probe_env}_samples={args.n_probe_samples}"
        )
        args.log_dir += f"_epochs={args.n_epochs}_noise={args.noise}"
    writer = SummaryWriter(args.log_dir)

    # log args
    pprint(f"args: \n{args}")
    for arg in vars(args):
        writer.add_text(f"args/{str(arg)}", str(getattr(args, arg)))

    seed = np.random.randint(0, 500)
    env: MiniGridEnv = gym.make(args.probe_env, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)

    # visualize environment
    writer.add_image("probe_env/world", env.render(), dataformats="HWC")
    world_grid = obs["image"]
    print(f"Probe env: {args.probe_env}")

    # generate ground truth Q
    true_Q = q_value_iteration(
        args.probe_env,
        world_grid,
        args.probe_q_threshold,
        args.alpha,
        args.gamma,
        writer,
        probe_env=True,
    )
    reset_env(env, seed)
    writer.add_video(
        "probe_env/q_value_iter_policy", visualize_policy(env, true_Q, seed)
    )

    # generate q and feedback samples
    all_q = true_Q.flatten()
    all_q = all_q[np.nonzero(all_q)]  # remove obstacle/goal q values
    np.random.shuffle(all_q)
    all_feedback = generate_feedback(all_q, args.noise, args.hd_feedback)


    q_data, coverage = generate_samples(env, args, world_grid, true_Q)
    np.random.shuffle(q_data)
    feedback_data = generate_feedback(q_data, args.noise, args.hd_feedback)

    kl_div = kl_divergence(q_data, all_q)
    print(f"State x Action space coverage: {coverage:.4f}")
    print(f"KL Divergence between reward samples and ground truth: {kl_div:.4f}")

    # tensorboard logging
    writer.add_histogram("q/q_data", q_data)
    writer.add_histogram("q/all_q", all_q)
    writer.add_histogram("feedback/feedback_data", feedback_data)
    writer.add_histogram("feedback/all_feedback", all_feedback)
    writer.add_text("stats/coverage", str(coverage))
    writer.add_text("stats/q_kl_div", str(kl_div))

    # train q model and visualize predicted Q and policy
    model = train_model(q_data, feedback_data, all_q, all_feedback, args, writer)
    fig, vid_tensor = visualize_q_model_policy(env, model, true_Q, args, seed)
    writer.add_figure("probe_env/q_model_pred_Q", fig)
    writer.add_video("probe_env/q_model_policy", vid_tensor)

    for downstream_env in args.downstream_envs:
        seed = np.random.randint(0, 500)

        # initialize environment
        env: MiniGridEnv = gym.make(downstream_env, render_mode="rgb_array")
        env = FullyObsWrapper(env)
        obs = reset_env(env, seed)
        writer.add_image(f"{downstream_env}/world", env.render(), dataformats="HWC")
        world_grid = obs["image"]
        print(f"Downstream env: {downstream_env}")

        # get ground truth q function
        true_Q = q_value_iteration(
            downstream_env,
            world_grid,
            args.probe_q_threshold,
            args.alpha,
            args.gamma,
            writer,
        )

        q_learning_from_q_model(
            env, downstream_env, world_grid, model, args, writer, seed, true_Q=true_Q
        )
        q_learning_from_true_q(
            env, downstream_env, world_grid, args, writer, seed, true_Q=true_Q
        )
        q_learning_from_env_reward(env, downstream_env, args, writer, seed)
        q_learning_from_feedback_as_q(
            env, downstream_env, world_grid, args, writer, seed, true_Q=true_Q
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
    from time import sleep

    sleep(0.5)
