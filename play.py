import torch
from torch import device
import gym
import time


class Play:
    def __init__(self, env, agent, max_episode=10):
        self.env = env
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        for _ in range(self.max_episode):
            s = self.env.reset()
            done = False
            episode_reward = 0
            # x = input("Push any button to proceed...")
            for _ in range(self.env._max_episode_steps):
                action = self.agent.choose_action(s)
                s_, r, done, _ = self.env.step(action)
                episode_reward += r
                if done:
                    break
                s = s_
                self.env.render(mode="human")
                # time.sleep(0.03)
            print(f"episode reward:{episode_reward:3.3f}")
