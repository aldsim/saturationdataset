import torch
import torch.nn as nn


class ALDNet0(nn.Module):

    def __init__(self, N):
        self.N = N
        super().__init__()
        self.linear_1 = nn.Linear(self.N, 1)
        self.linear_2 = nn.Linear(1, 1)

    def forward(self, x, t):
        x2 = self.linear_1(x)+self.linear_2(t)
        return x2


class ALDNet1(nn.Module):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        super().__init__()
        self.linear_1 = nn.Linear(self.N, self.M)
        self.linear_2 = nn.Linear(1, self.M)
        self.linear_3 = nn.Linear(self.M, 1)


    def forward(self, x, t):
        x2 = self.linear_1(x)+self.linear_2(t)
        x2 = torch.relu(x2)
        return self.linear_3(x2)


class ALDNet2(nn.Module):

    def __init__(self, N, M, M2):
        self.N = N
        self.M = M
        self.M2 = M2
        super().__init__()
        self.linear_1 = nn.Linear(self.N, self.M)
        self.linear_2 = nn.Linear(1, self.M)
        self.linear_3 = nn.Linear(self.M, self.M2)
        self.linear_4 = nn.Linear(self.M2, 1)

    def forward(self, x, t):
        x2 = self.linear_1(x)+self.linear_2(t)
        x2 = torch.relu(x2)
        x3 = torch.relu(self.linear_3(x2))
        return self.linear_4(x3)
