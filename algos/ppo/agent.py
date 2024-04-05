import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch

class Agent(nn.Module):
    def __init__(self, input_size, action_size):
        super(Agent, self).__init__()
        self.action_size = action_size
        self.conv = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        input_shape = (input_size[0] + 3, input_size[1] + 3)
        
        self.critic = nn.Sequential(
            nn.Linear(64*221, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(64*221, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, action_size)
        )
    
    def get_value(self, x):
        if(len(x.shape) == 4):
            x = self.conv(x)
        else:
            x = self.conv(x.unsqueeze(0)) # self.conv(x.permute(0, 3, 1, 2))
        return self.critic(x)
    
    def get_action_and_value(self, x, epsilon, action=None):
        if(len(x.shape) == 4):
            x = self.conv(x)
        else:
            x = self.conv(x.unsqueeze(0)) # self.conv(x.permute(0, 3, 1, 2))
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if np.random.rand() < epsilon:
            action = torch.randint(0, self.action_size - 1, size=(x.shape[0],), device=x.device)
        else:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)