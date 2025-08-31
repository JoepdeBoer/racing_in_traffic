from time import sleep

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecVideoRecorder

from train2 import config


if __name__ == "__main__":
    # config['screen_width'] = 1000
    # config['screen_height'] = 1000

    env = gym.make("racetrack-large-v0", config=config, render_mode= "rgb_array")
    modelpath = "./models/stable2vip-31"
    model = PPO.load(modelpath, env=env, device ="cpu")
    # observation, info = env.reset()
    # episode_over = False
    # env.render()
    # actionlst =[]
    # while not episode_over:
    #     action, _ = model.predict(observation=observation)
    #     actionlst.append(action)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     episode_over = terminated or truncated
    #     if terminated:
    #         print(f"observation: {observation["LidarObservation"]}")
    # Record the video starting at the first step
    video_folder = "vids"
    video_length = 512
    # env = VecVideoRecorder(env, video_folder,
    #                        record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                        name_prefix="last_model")

    env = RecordVideo(
        env, video_folder="vids", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
