import numpy as np
from model import PolicyNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy


class SAC:
    def __init__(self, n_states, n_actions, memory_size, batch_size):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)

    def store(self, state, reward, done, action, next_state):
        state = from_numpy(state).float().to(self.device)
        reward = torch.Tensor([reward]).to(self.device)
        done = torch.Tensor([done]).to(self.device)
        action = torch.Tensor([action]).to(self.device)
        next_state = from_numpy(next_state).float().to(self.device)
        self.memory.add(state, reward, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        dones = torch.cat(batch.done).to(self.device)
        actions = torch.cat(batch.action).view(-1, 1).to(self.device)
        next_states = torch.cat(batch.state).view(self.batch_size, self.n_states).to(self.device)

        return states, rewards, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return 0
        else:
            batch = self.memory.sample(self.batch_size)
            states, rewards, dones, actions, next_states = self.unpack(batch)
            


    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        action = self.policy.sample(states)
        return action.detach().cpu().numpy()
