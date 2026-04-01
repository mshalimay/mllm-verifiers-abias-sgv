import torch
import torch.nn as nn
import torch.optim as optim


class DLM_RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        output_dim,
        obs_keys,
        rnn_horizon
    ):
        """
        Args:
        - input_dim (int): dimension of input obs
        - hidden_dim (int): dimension of hidden state
        - num_layers (int): number of hidden layers
        - output_dim (int): dimension of output action
        - obs_keys (list of str): keys of obs modalities
        - rnn_horizon (int): how many steps to run the RNN before resetting the hidden state
        """
        super(DLM_RNN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.obs_keys = obs_keys
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.hidden_state = None
        self.hidden_state_horizon = rnn_horizon
        self.hidden_state_counter = 0
        
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.to(self.device)
        self.epoch = 0

    def train_on_batch(self, batch, validate):
        """
        Args:
        - batch (dict): batch of data batch['obs'].shape = (B, T, Obs_dim)
                        and batch['actions'].shape = (B, T, Action_dim)
        - validate (bool): whether batch is for validation

        Returns:
        - loss_val (float): value of loss
        """
        loss_val = None
        obs = batch['obs']
        act = batch['actions']
        self.reset()
        self.hidden_state = self.initialize_hidden_state(obs.shape[0])

        if validate:
            with torch.no_grad():
                out, _ = self.lstm(obs, self.hidden_state)
                policy_act = self.mlp(out)
                loss_val = self.loss(policy_act, act)
        else:
            out, _ = self.lstm(obs, self.hidden_state)
            policy_act = self.mlp(out)
            loss_val = self.loss(policy_act, act)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
        return loss_val

    def initialize_hidden_state(self, batch_size):
        """
        Args:
        - batch_size (int): size of batch
        Returns:
        - (h_0, c_0) : initial hidden state and cell state for LSTM
        """
        h_0 = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size,
                          self.hidden_size).to(self.device)
        return h_0, c_0

    def get_action(self, obs):
        """
        Args:
        - obs (dict): dictionary of observations of shape (B, 1, Obs_dim)
        Returns:
        - action (torch.Tensor): action of shape (B, 1, Action_dim)
        """
        action = None
        if self.hidden_state_counter % self.hidden_state_horizon == 0:
            self.reset()
            self.hidden_state = self.initialize_hidden_state(obs.shape[0])
        with torch.no_grad():
            out, self.hidden_state = self.lstm(obs, self.hidden_state)
            action = self.mlp(out)
        self.hidden_state_counter += 1
        return action

    def is_plan_reset(self,):
        return self.hidden_state_counter % self.hidden_state_horizon == 0

    def reset(self):
        self.hidden_state = None
        self.hidden_state_counter = 0
        return

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def set_horizon(self, rnn_horizon):
        self.hidden_state_horizon = rnn_horizon

    def set_eval(self):
        self.eval()
