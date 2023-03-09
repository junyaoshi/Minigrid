import numpy as np
import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv


from utils import WALL, N_DIRS, ACTIONS, N_ACTIONS


def random_action():
    return np.random.choice(ACTIONS)


def compute_coverage(world_grid, visited):
    assert isinstance(visited, set)
    total = (
        len([cell for row in world_grid for cell in row if cell[0] != WALL])
        * N_DIRS
        * N_ACTIONS
    )
    return len(visited) / total


if __name__ == "__main__":
    from PIL import Image
    from minigrid.wrappers import FullyObsWrapper
    import numpy as np
    from tqdm.auto import tqdm

    np.set_printoptions(linewidth=200)

    env_name = "MiniGrid-Empty-8x8-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    obs, _ = env.reset()
    world_grid = obs["image"]
    img = env.render()
    img = Image.fromarray(img)
    img.save("obs.jpg")

    n_random_samples = 1000
    visited = set()

    # env.step(2)
    # visited.add((env.agent_pos, env.agent_dir, 2))
    # img = env.render()
    # img = Image.fromarray(img)
    # img.save("obs.jpg")

    for _ in tqdm(range(n_random_samples)):
        action = random_action()
        visited.add((env.agent_pos, env.agent_dir, action))
        env.step(action)

        if env.step_count == env.max_steps:
            env.reset()

    from pprint import pprint

    pprint(sorted(visited))

    print(f"Coverage: {compute_coverage(world_grid, visited):.4f}")
