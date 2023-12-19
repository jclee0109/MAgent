import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class QmixAgent(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions):
        super(QmixAgent, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.hidden_dim = hidden_dim

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc1(obs)) # torch.Size([64])
        h = self.gru(x, hidden_state)
        q = self.fc2(h)
        return q, h
    
    def get_random_action(self, obs):
        return np.random.randint(self.dim_action, size=(self.num_agent), dtype=np.int32)