import numpy as np
from model import PolicyNetwork, QvalueNetwork, ValueNetwork
import torch
from replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam


class SAC:
    def __init__(self, n_states, n_actions, memory_size, batch_size, gamma, alpha, lr):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.memory = Memory(memory_size=self.memory_size)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_network = PolicyNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.q_value_network1 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.q_value_network2 = QvalueNetwork(n_states=self.n_states, n_actions=self.n_actions).to(self.device)
        self.value_network = ValueNetwork(n_states=self.n_states).to(self.device)
        self.value_target_network = ValueNetwork(n_states=self.n_states).to(self.device)
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

        self.value_loss = torch.nn.MSELoss()
        self.q_value_loss = torch.nn.MSELoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.lr)
        self.q_value_opt = Adam(list(self.q_value_network1.parameters()) + list(self.q_value_network2.parameters()),
                                lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)

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

            # Calculating the value target
            q1 = self.q_value_network1(states, actions)
            q2 = self.q_value_network2(states, actions)
            q = torch.min(q1, q2)
            reparam_action, log_prob = self.policy_network.sample_or_likelihood(states)
            target_value = q.detach() - log_prob.detach()

            value = self.value_network(states)
            value_loss = self.value_loss(value, target_value)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = rewards + self.gamma * self.value_target_network(next_states) * (1 - dones)
            q_loss = self.q_value_loss(q, target_value)

            # Calculating the Policy target
            with torch.no_grad():
                q1 = self.q_value_network1(states, actions)
                q2 = self.q_value_network2(states, actions)
                q = torch.min(q1, q2)
            policy_loss = self.alpha * log_prob - q

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value_opt.zero_grad()
            q_loss.backward()
            self.q_value_opt.step()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)




    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()

    @staticmethod
    def soft_update_target_network(local_network, target_network):
        pass
