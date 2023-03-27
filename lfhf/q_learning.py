from pprint import pprint
from copy import deepcopy

from PIL import Image
import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm, trange
import torch
import matplotlib.pyplot as plt

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

from utils import (
    print_world_grid,
    print_dist_to_goal,
    get_agent_state,
    reset_env,
    get_agent_state,
    hwc2chw,
    reset_env,
    find_goal,
    compute_coverage,
    ArrayAverageMeter,
    N_DIRS,
    N_ACTIONS,
    WALL,
    EAST,
    SOUTH,
    WEST,
    NORTH,
    GOAL,
    LEFT,
    RIGHT,
    FORWARD,
)
from reward import dijkstra, compute_reward
from behavior import random_action
from feedback import generate_feedback


EPSILON = 0.05
ALPHA = 0.1
GAMMA = 0.99


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


def choose_optimal_action(Q, state):
    """Returns the optimal action from Q-Value table.
    If multiple optimal actions, chooses random choice."""
    q_values_of_state = Q[state]
    max_q_value = np.max(q_values_of_state)
    action = np.random.choice(
        [a for a, q in enumerate(q_values_of_state) if q == max_q_value]
    )

    return action


def choose_action(Q, state, epsilon):
    """Returns the epsilon-greedy action based on Q table."""
    if np.random.uniform(0, 1) < epsilon:
        action = random_action()
    else:
        action = choose_optimal_action(Q, state)
    return action


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
        feedback = generate_feedback(true_reward, args.noise, args.hd_feedback)
        if args.hd_feedback == "base2":
            feedback = torch.tensor([feedback], device=args.device).float()
        else:
            feedback = torch.tensor([[feedback]], device=args.device).float()
        pred_reward = model(feedback).item()
        bellman_update(Q, state, pred_reward, new_state, action, args.alpha, args.gamma)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_from_q_model_episode(
    env: MiniGridEnv, model, true_Q, Q: ArrayAverageMeter, episode, args, seed, visited
):
    """Perform one Q-Learning episode where q comes from learned q model"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q.avg, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        visited.add((state, action))

        # use q model to process feedback to get estimated Q
        true_q = true_Q[state[0], state[1], state[2], action]
        feedback = generate_feedback(true_q, args.noise, args.hd_feedback)
        if args.hd_feedback == "base2":
            feedback = torch.tensor([feedback], device=args.device).float()
        else:
            feedback = torch.tensor([[feedback]], device=args.device).float()
        pred_q = model(feedback).item()
        Q.update((state[0], state[1], state[2], action), pred_q)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon, visited


def q_learning_from_true_q_episode(
    env: MiniGridEnv, true_Q, Q, episode, args, seed, visited
):
    """Perform one Q-Learning episode where q is ground truth q"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        visited.add((state, action))

        # use true Q to update Q
        true_q = true_Q[state[0], state[1], state[2], action]
        Q[state[0], state[1], state[2], action] = true_q

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon, visited


def q_learning_from_true_reward_episode(
    env: MiniGridEnv, dist_to_goal, Q, episode, args, seed
):
    """Perform one Q-Learning episode
    where reward is ground truth dense reward"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        new_state = get_agent_state(env)

        # get true reward and use it directly as reward to update Q
        true_reward = compute_reward(dist_to_goal, state, new_state)
        bellman_update(Q, state, true_reward, new_state, action, args.alpha, args.gamma)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_from_feedback_episode(
    env: MiniGridEnv, dist_to_goal, Q, episode, args, seed
):
    """Perform one Q-Learning episode
    where reward comes directly from feedback"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        new_state = get_agent_state(env)

        # get noisy feedback and use it directly as reward to update Q
        true_reward = compute_reward(dist_to_goal, state, new_state)
        feedback = generate_feedback(true_reward, args.noise)
        bellman_update(Q, state, feedback, new_state, action, args.alpha, args.gamma)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_from_feedback_as_q_episode(
    env: MiniGridEnv, true_Q, Q: ArrayAverageMeter, episode, args, seed, visited
):
    """Perform one Q-Learning episode where feedback is used as Q"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q.avg, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        visited.add((state, action))

        # get noisy feedback and use it directly as Q
        true_q = true_Q[state[0], state[1], state[2], action]
        feedback = generate_feedback(true_q, args.noise)
        Q.update((state[0], state[1], state[2], action), feedback)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon, visited


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


def visualize_Q(Q):
    """Visualize Q-Value table."""
    fig, ax = plt.subplots()
    Q_2d = np.transpose(np.mean(np.max(Q, axis=3), axis=2))
    im = ax.matshow(Q_2d)
    fig.colorbar(im)
    return fig


def visualize_policy(env: MiniGridEnv, Q, seed):
    """Visualize policy extracted from a Q table."""
    vid_array = [hwc2chw(env.render())]
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Visualize Policy Step", leave=False)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_optimal_action(Q, state)
        *_, terminated, truncated, _ = env.step(action)
        vid_array.append(hwc2chw(env.render()))
        pbar.update(1)

    reset_env(env, seed)
    pbar.close()

    vid_array = np.stack(vid_array, axis=0)[np.newaxis, :]
    return torch.from_numpy(vid_array)


def visualize_q_model_policy(env: MiniGridEnv, model, Q, args, seed):
    """Visualize policy where Q table is predicted
    using Q model q=model(feedback)"""
    feedback = generate_feedback(Q, args.noise, args.hd_feedback)
    feedback = torch.from_numpy(feedback).to(args.device).float()
    print(feedback.shape)
    if args.hd_feedback == "base2":
        reshaped_feedback = feedback.view(-1, feedback.shape[-1])
        pred_Q = model(reshaped_feedback)
        print(reshaped_feedback.shape)
    else:
        pred_Q = model(feedback.flatten().unsqueeze(1))

    pred_Q = pred_Q.detach().cpu().numpy().reshape(Q.shape)
    pred_Q[Q == 0] = 0
    fig = visualize_Q(pred_Q)

    vid_array = [hwc2chw(env.render())]
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Visualize Policy Step", leave=False)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_optimal_action(pred_Q, state)
        *_, terminated, truncated, _ = env.step(action)
        vid_array.append(hwc2chw(env.render()))
        pbar.update(1)

    reset_env(env, seed)
    pbar.close()

    vid_array = np.stack(vid_array, axis=0)[np.newaxis, :]
    return fig, torch.from_numpy(vid_array)


def q_learning_from_reward_model(env_name, model, args, writer, seed):
    """Q-Learning where reward comes from learned reward model
    r = model(feedback)"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_reward_model_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print("Learning from reward model")
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

        writer.add_scalar(
            f"{env_name}/learn_from_reward_model_epsilon", epsilon, episode
        )
        writer.add_scalar(
            f"{env_name}/learn_from_reward_model_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_figure(f"{env_name}/learn_from_reward_model_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_reward_model_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_true_reward(env_name, args, writer, seed):
    """Q-Learning where reward comes from ground truth dense reward"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_true_reward_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print("Learning directly from true reward")
    print(f"Downstream env: {env_name}")
    print_world_grid(world_grid)

    # get ground truth dense reward function
    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    for episode in tqdm(
        range(args.n_episodes), desc="Q-Learning from True Reward Episode"
    ):
        epsilon = q_learning_from_true_reward_episode(
            env, dist_to_goal, Q, episode, args, seed
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)

        writer.add_scalar(
            f"{env_name}/learn_from_true_reward_epsilon", epsilon, episode
        )
        writer.add_scalar(
            f"{env_name}/learn_from_true_reward_reward", cumulative_env_reward, episode
        )
        writer.add_figure(f"{env_name}/learn_from_true_reward_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_true_reward_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_feedback_as_reward(env_name, args, writer, seed):
    """Q-Learning where feedback is directly used as reward"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_feedback_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print("Leanring directly from feedback")
    print(f"Downstream env: {env_name}")
    print_world_grid(world_grid)

    # get ground truth dense reward function
    dist_to_goal = dijkstra(world_grid)
    print_dist_to_goal(dist_to_goal)

    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    for episode in tqdm(
        range(args.n_episodes), desc="Q-Learning from Feedback Episode"
    ):
        epsilon = q_learning_from_feedback_episode(
            env, dist_to_goal, Q, episode, args, seed
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

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_feedback_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_env_reward(env, env_name, args, writer, seed):
    """Q-Learning where reward comes from environment"""
    print("Learning from environment reward")

    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    pbar = trange(args.n_episodes)
    for episode in pbar:
        pbar.set_description(f"Episode={episode + 1}")
        epsilon = q_learning_from_env_reward_episode(env, Q, episode, args, seed)
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)

        pbar.set_postfix({"rew": f"{cumulative_env_reward:.2f}"})
        writer.add_scalar(f"{env_name}/learn_from_env_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_env_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_figure(f"{env_name}/learn_from_env_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_env_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_q_model(
    env, env_name, world_grid, model, args, writer, seed, true_Q=None
):
    """Q-Learning where reward comes from learned q model
    q = model(feedback)"""
    print("Learning from Q model")

    # get ground truth Q
    if true_Q is None:
        true_Q = q_value_iteration(
            env_name, world_grid, args.probe_q_threshold, args.alpha, args.gamma, writer
        )

    visited = set()
    Q = ArrayAverageMeter((env.width, env.height, N_DIRS, N_ACTIONS))
    pbar = trange(args.n_episodes)
    for episode in pbar:
        pbar.set_description(f"Episode={episode + 1}")
        epsilon, visited = q_learning_from_q_model_episode(
            env, model, true_Q, Q, episode, args, seed, visited
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q.avg, seed)
        fig = visualize_Q(Q.avg)
        q_coverage = compute_coverage(world_grid, visited)

        pbar.set_postfix(
            {"q_cover": f"{q_coverage:.2f}", "rew": f"{cumulative_env_reward:.2f}"}
        )
        writer.add_scalar(f"{env_name}/learn_from_q_model_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_q_model_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_scalar(
            f"{env_name}/learn_from_q_model_coverage", q_coverage, episode
        )
        writer.add_figure(f"{env_name}/learn_from_q_model_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_q_model_policy", visualize_policy(env, Q.avg, seed)
    )


def q_learning_from_true_q(env, env_name, world_grid, args, writer, seed, true_Q=None):
    """Q-Learning where reward comes from ground truth Q table"""
    print("Learning from true Q")

    # get ground truth Q
    if true_Q is None:
        true_Q = q_value_iteration(
            env_name, world_grid, args.probe_q_threshold, args.alpha, args.gamma, writer
        )

    visited = set()
    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    pbar = trange(args.n_episodes)
    for episode in pbar:
        pbar.set_description(f"Episode={episode + 1}")
        epsilon, visited = q_learning_from_true_q_episode(
            env, true_Q, Q, episode, args, seed, visited
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)
        q_coverage = compute_coverage(world_grid, visited)

        pbar.set_postfix(
            {"q_cover": f"{q_coverage:.2f}", "rew": f"{cumulative_env_reward:.2f}"}
        )
        writer.add_scalar(f"{env_name}/learn_from_true_q_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_true_q_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_scalar(f"{env_name}/learn_from_true_q_coverage", q_coverage, episode)
        writer.add_figure(f"{env_name}/learn_from_true_q_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_true_q_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_feedback_as_q(
    env, env_name, world_grid, args, writer, seed, true_Q=None
):
    """Q-Learning where feedback is directly used as q"""
    print("Learning directly from feedback as Q")

    # get ground truth Q
    if true_Q is None:
        true_Q = q_value_iteration(
            env_name, world_grid, args.probe_q_threshold, args.alpha, args.gamma, writer
        )

    visited = set()
    Q = ArrayAverageMeter((env.width, env.height, N_DIRS, N_ACTIONS))
    pbar = trange(args.n_episodes)
    for episode in pbar:
        pbar.set_description(f"Episode={episode + 1}")
        epsilon, visited = q_learning_from_feedback_as_q_episode(
            env, true_Q, Q, episode, args, seed, visited
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q.avg, seed)
        fig = visualize_Q(Q.avg)
        q_coverage = compute_coverage(world_grid, visited)

        pbar.set_postfix(
            {"q_cover": f"{q_coverage:.2f}", "rew": f"{cumulative_env_reward:.2f}"}
        )
        writer.add_scalar(
            f"{env_name}/learn_from_feedback_as_q_epsilon", epsilon, episode
        )
        writer.add_scalar(
            f"{env_name}/learn_from_feedback_as_q_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_scalar(
            f"{env_name}/learn_from_feedback_as_q_coverage", q_coverage, episode
        )
        writer.add_figure(f"{env_name}/learn_from_feedback_as_q_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_feedback_as_q_policy", visualize_policy(env, Q.avg, seed)
    )


def update_Q(world_grid, Q, R, alpha, gamma):
    """Perform one iteration of the Q-value Iteration algorithm"""
    width, height, _ = world_grid.shape
    new_Q = deepcopy(Q)
    for x in range(width):
        for y in range(height):
            for dir in range(N_DIRS):
                for action in range(N_ACTIONS):
                    if world_grid[x, y, 0] == WALL or world_grid[x, y, 0] == GOAL:
                        continue
                    new_x, new_y, new_dir = step(world_grid, (x, y, dir), action)

                    q_values_of_state = Q[new_x, new_y, new_dir, :]
                    max_q_value_in_new_state = max(q_values_of_state)
                    current_q_value = Q[x, y, dir, action]
                    new_Q[x, y, dir, action] = (1 - alpha) * current_q_value + alpha * (
                        R[x, y, dir, action] + gamma * max_q_value_in_new_state
                    )

    return new_Q


def q_value_iteration(
    env_name, world_grid, q_threshold, alpha, gamma, writer, probe_env=False
):
    """Performs Q-Value Iteration algorithm, similar to Value Iteration.
    Returns a Q-Value table Q: S x A -> scalar for the given world grid."""
    if probe_env:
        env_name = "probe_env"
    width, height, _ = world_grid.shape
    Q = np.zeros((width, height, N_DIRS, N_ACTIONS))
    R = generate_reward_table(world_grid)

    pbar = trange(10000)
    q_diff = np.inf
    step = 0
    while q_diff > q_threshold:
        pbar.set_description(f"Q-Value iteration step={step + 1}")
        new_Q = update_Q(world_grid, Q, R, alpha, gamma)
        q_diff = np.sum(np.absolute(new_Q - Q))
        Q = new_Q

        step += 1
        pbar.update(1)
        pbar.set_postfix({"q_diff": f"{q_diff:.5f}"})
        writer.add_scalar(f"{env_name}/q_value_iter_diff", q_diff, step)
        if step % 10 == 0:
            fig = visualize_Q(Q)
            writer.add_figure(f"{env_name}/Q_value_iter", fig, step)

    pbar.close()
    fig = visualize_Q(Q)
    writer.add_figure(f"{env_name}/Q_value_iter", fig, step)

    return Q


def generate_reward_table(world_grid):
    """Returns a reward table R: S x A -> {0, 1} for the given world grid."""
    width, height, _ = world_grid.shape
    R = np.zeros((width, height, N_DIRS, N_ACTIONS))
    goal_x, goal_y = find_goal(world_grid)
    assert goal_x != 0 and goal_y != 0 and goal_x != width - 1 and goal_y != height - 1
    if world_grid[goal_x - 1, goal_y, 0] != WALL:
        R[goal_x - 1, goal_y, EAST, FORWARD] = 1
    if world_grid[goal_x + 1, goal_y, 0] != WALL:
        R[goal_x + 1, goal_y, WEST, FORWARD] = 1
    if world_grid[goal_x, goal_y - 1, 0] != WALL:
        R[goal_x, goal_y - 1, SOUTH, FORWARD] = 1
    if world_grid[goal_x, goal_y + 1, 0] != WALL:
        R[goal_x, goal_y + 1, NORTH, FORWARD] = 1

    return R


def step(world_grid, state, action):
    width, height, _ = world_grid.shape
    x, y, dir = state
    new_x = x
    new_y = y
    new_dir = dir

    # Rotate left
    if action == LEFT:
        new_dir -= 1
        if new_dir < 0:
            new_dir += 4

    # Rotate right
    elif action == RIGHT:
        new_dir = (new_dir + 1) % 4

    # Move forward
    elif action == FORWARD:
        if dir == EAST and world_grid[x + 1, y, 0] != WALL:
            new_x += 1
        if dir == SOUTH and world_grid[x, y + 1, 0] != WALL:
            new_y += 1
        if dir == WEST and world_grid[x - 1, y, 0] != WALL:
            new_x -= 1
        if dir == NORTH and world_grid[x, y - 1, 0] != WALL:
            new_y -= 1

    return new_x, new_y, new_dir


if __name__ == "__main__":
    # Q-Learning example
    from utils import find_agent

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
