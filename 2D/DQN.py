import json
import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):

    def __init__(self, num_actions, memory_cap=25, epsilon=0.1, batch_size=16, gamma=0.1):
        super(DQN, self).__init__()
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3, 16, 8, 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.linear1 = nn.Linear(3520, 50)
        self.gamma = gamma
        self.memory_size = memory_cap
        self.deque = deque(maxlen=memory_cap)
        self.linear2 = nn.Linear(50, num_actions)

    def add_to_deque(self, state, action, reward, next_state, done):
        self.deque.append((state, action, reward, next_state, done))

    def pick_action(self, state):
        # implementing epsilon-greedy method of exploration vs exploitation
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return torch.argmax(self.forward(state))

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        out = self.bn1(F.relu(self.conv1(x)))
        out = self.bn2(F.relu(self.conv2(out)))
        out = F.relu(self.linear1(out.view(1, -1)))
        score = self.linear2(out)
        return score

    def replay_experience(self, state, action, reward, done, next_state, optimizer, scheduler):
        if len(self.deque) < self.batch_size:
            return
        loss = nn.MSELoss()
        mini_batch = random.sample(self.deque, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            if not done:
                target = self.gamma * torch.max(self.forward(next_state))[0]
                target = target + torch.FloatTensor([reward])
                target_present = self.forward(state)
                target_present = target_present[:, action]
                target = target.detach()
                output_loss = loss(target, target_present)
                optimizer.zero_grad()
                output_loss.backward()
                output_loss = output_loss.item()
                scheduler.step()
                print(output_loss)
                optimizer.step()


def train():
    dqn = DQN(11)
    if torch.cuda.is_available():
        dqn = dqn.cuda()

    try:
        with open('state_action_map.json') as f:
            state_action_map_dict = json.load(f)
    except Exception as e:
        print(e)
    dataset = []
    optimizer = torch.optim.RMSprop(dqn.parameters(), lr=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, 0.1)

    num_demos = 25
    for i in range(num_demos):
        print("Epoch")
        print(i)
        for file_id in state_action_map_dict.keys():
            print("File")
            print(file_id)
            for key in state_action_map_dict[file_id]:
                state = np.asarray(
                    cv2.resize(cv2.imread("screen_capture/" + str(file_id) + "/" + str(int(key) - 1) + ".png"),
                               (100, 95))).reshape(3, 100, 95) / 255
                next_state = np.asarray(
                    cv2.resize(cv2.imread("screen_capture/" + str(file_id) + "/" + str(int(key)) + ".png"),
                               (100, 95))).reshape(3, 100, 95) / 255
                action = state_action_map_dict[file_id][key][0]
                reward = state_action_map_dict[file_id][key][1]
                done = state_action_map_dict[file_id][key][2]
                dqn.add_to_deque(torch.FloatTensor([state]), action, reward, torch.FloatTensor([next_state]), done)
                dqn.replay_experience(state, action, reward, done, next_state, optimizer, lr_scheduler)
    torch.save(dqn.state_dict(), "dqn.pth")


if __name__ == "__main__":
    train()
