import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the mixing network for the global Q value in the multi-agent setting
class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_dim = mixing_dim

        self.hyper_w_1 = nn.Linear(self.state_dim, self.mixing_dim * self.n_agents)
        self.hyper_w_final = nn.Linear(self.state_dim, self.mixing_dim)
        
        self.hyper_b_1 = nn.Linear(state_dim, mixing_dim)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(state_dim, mixing_dim),
            nn.ReLU(),
            nn.Linear(mixing_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents) # bs * 1 * n_agents
        # First layer
        w1 = torch.abs(self.hyper_w_1(states)) # mixing_dim * n_agents
        b1 = self.hyper_b_1(states) # mixing_dim
        w1 = w1.view(-1, self.n_agents, self.mixing_dim) # bs * n_agents * mixing_dim
        b1 = b1.view(-1, 1, self.mixing_dim) # bs * 1 * mixing_dim
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1) # bs * 1 * mixing_dim
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states)) # mixing_dim
        w_final = w_final.view(-1, self.mixing_dim, 1) # bs * mixing_dim * 1
        # State-dependent bias
        v = self.hyper_b_final(states).view(-1, 1, 1) # bs * 1 * 1
        # Compute final output
        y = torch.bmm(hidden, w_final) + v # bs * 1 * 1
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

# Define a basic GNN layer
class GraphMixLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphMixLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj_matrix):
        # Apply the linear transformation
        h = self.linear(x)
        # Multiply by adjacency matrix to aggregate neighbor information
        h = torch.matmul(adj_matrix, h)
        return F.relu(h)

# Define the centralized mixing GNN
class CentralizedMixingGNN(nn.Module):
    def __init__(self, n_agents, state_dim, mixing_dim, readout_dim):
        super(CentralizedMixingGNN, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.mixing_dim = mixing_dim
        self.readout_dim = readout_dim
        
        # Define multiple layers of GNN
        self.gnn_layers = nn.ModuleList([
            GraphMixLayer(state_dim, mixing_dim) for _ in range(n_agents)
        ])
        self.readout_layer = nn.Linear(mixing_dim, readout_dim)

    def forward(self, agent_qs, states, adj_matrix):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents, 1)
        
        # Process each agent's Qs through the GNN layers
        for gnn_layer in self.gnn_layers:
            agent_qs = gnn_layer(agent_qs, adj_matrix)

        # Apply the readout layer to get the global Q value
        q_total = self.readout_layer(agent_qs)
        q_total = q_total.view(bs, -1, self.readout_dim)

        return q_total