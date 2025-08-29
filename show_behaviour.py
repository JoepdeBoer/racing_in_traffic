from time import sleep
from stable_baselines3 import PPO
import gymnasium as gym
import matplotlib.pyplot as plt
from train import config


if __name__ == "__main__":

    env = gym.make("racetrack-large-v0", config=config, render_mode="human")
    modelpath = "./models/racecar-25"
    model = PPO.load(modelpath, env=env, device ="cpu")
    observation, info = env.reset()
    episode_over = False
    env.render()
    actionlst =[]
    while not episode_over:
        action, _ = model.predict(observation=observation)
        actionlst.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated

    env.close()

    plt.plot(actionlst)
    plt.show()