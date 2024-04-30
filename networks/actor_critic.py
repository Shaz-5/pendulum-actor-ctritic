import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, seed=42, hidden_dim1=512, hidden_dim2=256):
        """
        Initialize Actor Network.
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_dim, hidden_dim1)
        self.hidden_layer_1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, action_dim)

    def forward(self, state):
        """
        Build Actor that maps states to actions.
        """
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer_1(x))
        return F.tanh(self.output_layer(x))
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, seed=42, hidden_dim1=512, hidden_dim2=256):
        """
        Initialize Critic Network.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_dim, hidden_dim1)
        self.hidden_layer_1 = nn.Linear(hidden_dim1+action_dim, hidden_dim2)
        self.output_layer = nn.Linear(hidden_dim2, 1)

    def forward(self, state, action):
        """
        Build Critic that maps (state, action) pairs to Q values.
        """
        x = F.relu(self.input_layer(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.hidden_layer_1(x))
        return self.output_layer(x)