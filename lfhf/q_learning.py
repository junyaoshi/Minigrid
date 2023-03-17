from pprint import pprint
from PIL import Image

import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.utils.window import Window

from utils import print_world_grid, print_dist_to_goal, get_agent_state, reset_env
from utils import N_DIRS, N_ACTIONS
from reward import dijkstra, compute_reward
from behavior import random_action
from feedback import noisy_sigmoid_feedback

EPSILON = 0.05
ALPHA = 0.1
GAMMA = 0.99


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


def choose_action(Q, state, epsilon):
    """Returns the epsilon-greedy action."""
    if np.random.uniform(0, 1) < epsilon:
        action = random_action()
    else:
        action = choose_optimal_action(Q, state)
    return action


def choose_optimal_action(Q, state):
    """Returns the optimal action from Q-Value table.
    If multiple optimal actions, chooses random choice."""
    q_values_of_state = Q[state]
    max_q_value = np.max(q_values_of_state)
    action = np.random.choice(
        [a for a, q in enumerate(q_values_of_state) if q == max_q_value]
    )

    return action


def choose_epsilon(args, episode):
    """Returns epsilon for epsilon-greedy action selection."""
    epsilon = args.epsilon
    if args.epsilon_linear_decay:
        epsilon = (
            args.epsilon_min - args.epsilon
        ) / args.n_episodes * episode + args.epsilon
        epsilon = max(args.epsilon_min, epsilon)
    return epsilon


def bellman_update(Q, state, reward, new_state, action, alpha, gamma):
    """Updates the Q-value table using the Bellman equation."""
    q_values_in_new_state = Q[new_state]
    max_q_value_in_new_state = np.max(q_values_in_new_state)
    current_q_value = Q[state][action]

    Q[state][action] = (1 - alpha) * current_q_value + alpha * (
        reward + gamma * max_q_value_in_new_state
    )


def visualize_Q(Q):
    """Visualize Q-Value table."""
    fig, ax = plt.subplots()
    Q_2d = np.transpose(np.mean(np.max(Q, axis=3), axis=2))
    im = ax.matshow(Q_2d)
    fig.colorbar(im)
    # for i in range(Q_2d.shape[0]):
    #     for j in range(Q_2d.shape[1]):
    #         ax.text(i, j, str(round(Q_2d[i, j])), va='center', ha='center', fontsize=5)

    return fig


def q_learning_from_reward_model_episode(
    env: MiniGridEnv, model, dist_to_goal, Q, episode, args, seed
):
    """Perform one Q-Learning episode
    where reward comes from learned reward model"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        new_state = get_agent_state(env)

        # use reward model to process feedback to get estimated reward and update Q
        true_reward = compute_reward(dist_to_goal, state, new_state)
        feedback = noisy_sigmoid_feedback(true_reward, args.noise)
        feedback = torch.tensor([[feedback]], device=args.device).float()
        pred_reward = model(feedback).item()
        bellman_update(Q, state, pred_reward, new_state, action, args.alpha, args.gamma)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_from_env_reward_episode(env: MiniGridEnv, Q, episode, args, seed):
    """Perform one Q-Learning episode
    where reward comes from environment"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        _, env_reward, terminated, truncated, _ = env.step(action)
        new_state = get_agent_state(env)

        # use env reward to update Q
        bellman_update(Q, state, env_reward, new_state, action, args.alpha, args.gamma)
        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_eval_episode(env: MiniGridEnv, Q, seed):
    """Perform one Q-Learning evaluation episode"""
    terminated, truncated = False, False
    cumulative_env_reward = 0
    pbar = tqdm(total=env.max_steps, desc="Eval Q-Learning Step", leave=False)

    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_optimal_action(Q, state)
        _, env_reward, terminated, truncated, _ = env.step(action)

        cumulative_env_reward += env_reward
        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return cumulative_env_reward


def q_learning_from_reward_model(env_name, model, args, writer, seed):
    """Q-Learning where reward comes from learned reward model
    r = model(feedback)"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_feedback_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print(f"Downstream env: {env_name}")
    print_world_grid(world_grid)

    # get ground truth dense reward function
    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    for episode in tqdm(
        range(args.n_episodes), desc="Q-Learning from Reward Model Episode"
    ):
        epsilon = q_learning_from_reward_model_episode(
            env, model, dist_to_goal, Q, episode, args, seed
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)

        writer.add_scalar(f"{env_name}/learn_from_feedback_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_feedback_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_figure(f"{env_name}/learn_from_feedback_Q", fig, episode)
        plt.close(fig)


def q_learning_from_env_reward(env_name, args, writer, seed):
    """Q-Learning where reward comes from environment"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    writer.add_image(
        f"{env_name}/learn_from_env_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print(f"Downstream env: {env_name}")
    print_world_grid(world_grid)

    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    for episode in tqdm(
        range(args.n_episodes), desc="Q-Learning from Env Reward Episode"
    ):
        epsilon = q_learning_from_env_reward_episode(env, Q, episode, args, seed)
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)

        writer.add_scalar(f"{env_name}/learn_from_env_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_env_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_figure(f"{env_name}/learn_from_env_Q", fig, episode)
        plt.close(fig)


if __name__ == "__main__":
    # Q-Learning example
    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-Empty-Random-5x5-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    world_grid = obs["image"]
    print("World grid:")
    pprint(np.transpose(world_grid[:, :, 0]))
    pprint(np.transpose(world_grid[:, :, 1]))
    pprint(np.transpose(world_grid[:, :, 2]))

    q_table = dict()
    for x in range(env.width):
        for y in range(env.height):
            for dir in range(4):
                q_table[(x, y, dir)] = {
                    env.actions.left: 0,
                    env.actions.right: 0,
                    env.actions.forward: 0,
                }
    available_actions = [env.actions.left, env.actions.right, env.actions.forward]

    num_trials = 1000
    reward_per_episode = []
    for trial in range(num_trials):
        terminated = False
        truncated = False
        print(find_agent(world_grid))
        cumulative_reward = 0
        step = 0
        while not (terminated or truncated):
            old_state = find_agent(world_grid)
            action = choose_action(q_table, world_grid, available_actions, EPSILON)
            obs, reward, terminated, truncated, _ = env.step(action)
            world_grid = obs["image"]
            new_state = find_agent(world_grid)
            bellman_update(q_table, old_state, reward, new_state, action, ALPHA, GAMMA)
            cumulative_reward += reward
            step += 1
        obs, _ = env.reset()
        world_grid = obs["image"]
        reward_per_episode.append(cumulative_reward)
    print(reward_per_episode)
    plt.plot(reward_per_episode)
    plt.show()

    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        world_grid = obs["image"]
        state = find_agent(world_grid)
        print(q_table[state])
        img = env.render()
        img = Image.fromarray(img)
        img.save("obs.jpg")
        next = input("Press ENTER to go next: ")
        action = choose_action(q_table, world_grid, available_actions, 0)
        print(action)
        obs, reward, terminated, truncated, _ = env.step(action)
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            for dir in range(4):
                print(x, y, dir, q_table[(x, y, dir)])
    print("done")