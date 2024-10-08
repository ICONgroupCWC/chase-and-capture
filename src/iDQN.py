import torch.nn as nn


class iQNetworkCoordinated(nn.Module):
    def __init__(self, m_agents, p_preys, action_size, agent_id):
        super(iQNetworkCoordinated, self).__init__()

        assert agent_id != 0
        state_dims = (m_agents + p_preys) * 2 + p_preys
        input_dims = state_dims

        self.net = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, action_size*(agent_id)),
        )

    def forward(self, x):
        x = self.net(x)
        return x



class iQNetworkMCTS(nn.Module):
    def __init__(self, m_agents, p_preys, action_size, agent_id):
        super(iQNetworkCoordinated, self).__init__()

        assert agent_id != 0
        state_dims = (m_agents + p_preys) * 2 + p_preys
        input_dims = state_dims

        self.net = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, action_size*(agent_id)),
        )

    def forward(self, x):
        x = self.net(x)
        return x