import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, num_actions, memory_cap=2000, epsilon=0.1, batch_size=256, gamma=0.1):
        super(DQN, self).__init__()
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3, 16, 8, 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(32, 256)
        self.gamma = gamma
        self.memory_size = memory_cap
        self.deque = deque(maxlen=memory_cap)
        self.linear2 = nn.Linear(256, num_actions)

    def add_to_deque(self, state, action, reward, next_state, done):
        self.deque.append((state, action, reward, next_state, done))

    def pick_action(self, state):
        # implementing epsilon-greedy method of exploration vs exploitation
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return torch.argmax(self.forward(state))

    def forward(self, *input):
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = F.relu(self.linear1(out))
        score = self.linear2(out)
        return score

    def replay_experience(self):
        if len(deque) < self.batch_size:
            return
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        mini_batch = random.sample(self.deque, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.forward(next_state))
            target_present = self.forward(state)
            target_present[action] = target
            output_loss = loss(target_present, target)
            optimizer.zero_grad()
            output_loss.backward()
            optimizer.step()


def train():
    dqn = DQN(6)
    dataset = []
    num_demos = 50
    for i in range(num_demos):
        data = dataset[i]
        for j in range(0, len(dataset[i])):
            # dqn.add_to_deque()
            dqn.replay_experience()
            pass
