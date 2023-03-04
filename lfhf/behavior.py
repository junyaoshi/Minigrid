import gymnasium as gym

from minigrid.minigrid_env import MiniGridEnv


def random_action(env: MiniGridEnv):
    return env.action_space.sample()


if __name__ == "__main__":
    env_name = "MiniGrid-FourRooms-v0"
    env: MiniGridEnv = gym.make(env_name, render_mode="rgb_array")
