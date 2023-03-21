from pprint import pprint
from copy import deepcopy

from PIL import Image
import numpy as np
import gymnasium as gym
from tqdm.auto import tqdm
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
from feedback import noisy_sigmoid_feedback


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
        feedback = noisy_sigmoid_feedback(true_reward, args.noise)
        feedback = torch.tensor([[feedback]], device=args.device).float()
        pred_reward = model(feedback).item()
        bellman_update(Q, state, pred_reward, new_state, action, args.alpha, args.gamma)

        pbar.update(1)

    reset_env(env, seed)
    pbar.close()
    return epsilon


def q_learning_from_q_model_episode(
    env: MiniGridEnv, model, q_ground_truth, Q, episode, args, seed, visited
):
    """Perform one Q-Learning episode
    where q comes from learned q model"""
    terminated, truncated = False, False
    pbar = tqdm(total=env.max_steps, desc="Q-Learning Step", leave=False)

    epsilon = choose_epsilon(args, episode)
    while not (terminated or truncated):
        state = get_agent_state(env)
        action = choose_action(Q, state, epsilon)
        *_, terminated, truncated, _ = env.step(action)
        visited.add((state, action))

        # use q model to process feedback to get estimated Q
        true_q = q_ground_truth[state[0], state[1], state[2], action]
        feedback = noisy_sigmoid_feedback(true_q, args.noise)
        feedback = torch.tensor([[feedback]], device=args.device).float()
        pred_q = model(feedback).item()
        Q[state[0], state[1], state[2], action] = pred_q

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
        feedback = noisy_sigmoid_feedback(true_reward, args.noise)
        bellman_update(Q, state, feedback, new_state, action, args.alpha, args.gamma)

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


def visualize_Q(Q):
    """Visualize Q-Value table."""
    fig, ax = plt.subplots()
    Q_2d = np.transpose(np.mean(np.max(Q, axis=3), axis=2))
    im = ax.matshow(Q_2d)
    fig.colorbar(im)
    return fig


def visualize_policy(env: MiniGridEnv, Q, seed):
    """Visualize policy."""
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
        f"{env_name}/learn_from_true_world", env.render(), dataformats="HWC"
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

        writer.add_scalar(f"{env_name}/learn_from_true_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_true_reward", cumulative_env_reward, episode
        )
        writer.add_figure(f"{env_name}/learn_from_true_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_true_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_feedback(env_name, args, writer, seed):
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


def q_learning_from_env_reward(env_name, args, writer, seed):
    """Q-Learning where reward comes from environment"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_env_world", env.render(), dataformats="HWC"
    )
    print("Learning from environment reward")
    print(f"Downstream env: {env_name}")

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

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_env_policy", visualize_policy(env, Q, seed)
    )


def q_learning_from_q_model(env_name, model, args, writer, seed):
    """Q-Learning where reward comes from learned q model
    q = model(feedback)"""
    # initialize environment
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs = reset_env(env, seed)
    writer.add_image(
        f"{env_name}/learn_from_q_model_world", env.render(), dataformats="HWC"
    )
    world_grid = obs["image"]
    print("Learning from Q model")
    print(f"Downstream env: {env_name}")

    # get ground truth q function
    q_table = q_value_iteration(
        env_name, world_grid, args.probe_q_threshold, args.alpha, args.gamma, writer
    )
    visited = set()
    Q = np.zeros((env.width, env.height, N_DIRS, N_ACTIONS))
    for episode in tqdm(
        range(args.n_episodes), desc=f"Q-Learning from Q Model Episode"
    ):
        epsilon, visited = q_learning_from_q_model_episode(
            env, model, q_table, Q, episode, args, seed, visited
        )
        cumulative_env_reward = q_learning_eval_episode(env, Q, seed)
        fig = visualize_Q(Q)
        q_coverage = compute_coverage(world_grid, visited)

        writer.add_scalar(f"{env_name}/learn_from_q_model_epsilon", epsilon, episode)
        writer.add_scalar(
            f"{env_name}/learn_from_q_model_reward",
            cumulative_env_reward,
            episode,
        )
        writer.add_scalar(f"{env_name}/learn_from_q_model_coverage", q_coverage, episode)
        writer.add_figure(f"{env_name}/learn_from_q_model_Q", fig, episode)
        plt.close(fig)

    # visualize policy
    writer.add_video(
        f"{env_name}/learn_from_q_model_policy", visualize_policy(env, Q, seed)
    )


def update_Q(world_grid, q_table, reward_table, width, height, alpha, gamma):
    new_q_table = deepcopy(q_table)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            for dir in range(N_DIRS):
                for action in range(N_ACTIONS):
                    if world_grid[x, y, 0] == WALL or world_grid[x, y, 0] == GOAL:
                        continue
                    new_x, new_y, new_dir = step(world_grid, (x, y, dir), action)

                    q_values_of_state = q_table[new_x, new_y, new_dir, :]
                    max_q_value_in_new_state = max(q_values_of_state)
                    current_q_value = q_table[x, y, dir, action]
                    new_q_table[x, y, dir, action] = (
                        1 - alpha
                    ) * current_q_value + alpha * (
                        reward_table[x, y, dir, action]
                        + gamma * max_q_value_in_new_state
                    )

    return new_q_table


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

    pbar = tqdm(desc="Q-Value Iteration", total=10000)
    q_diff = np.inf
    step = 0
    while q_diff > q_threshold:
        new_Q = update_Q(world_grid, Q, R, width, height, alpha, gamma)
        q_diff = np.sum(np.absolute(new_Q - Q))
        Q = new_Q

        step += 1
        pbar.update(1)
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
