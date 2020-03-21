import numpy as np
from model import PolicyNetwork
import torch
from replay_memory import Memory


class SAC:
    def __init__(self, n_states, n_actions, memory_size):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_size = memory_size
        memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)





    def store(self, state, reward, done, action, next_state):
        pass

    def train(self):
        pass



    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        action = self.policy.sample(states)
        return action.detach().cpu().numpy()