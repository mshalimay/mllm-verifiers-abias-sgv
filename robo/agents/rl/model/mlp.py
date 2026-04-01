import torch
import torch.nn as nn


class DLMMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        obs_keys
    ):
        """
        Args:
        - input_dim (int): dimension of input obs
        - hidden_dims (list of int): dimensions of hidden layers
        - output_dim (int): dimension of output action
        - obs_keys (list of str): keys of obs modalities
        """
        super(DLMMLP, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = nn.ModuleList()
        self.actor.append(nn.Linear(input_dim, hidden_dims[0]))
        for k in range(len(hidden_dims)-1):
            self.actor.append(nn.LeakyReLU())
            self.actor.append(nn.Linear(hidden_dims[k], hidden_dims[k+1]))
        self.actor.append(nn.LeakyReLU())
        self.actor.append(nn.Linear(hidden_dims[-1], output_dim))
        self.optimizer = torch.optim.Adam(
            params=self.actor.parameters(), lr=0.0001,weight_decay=1e-4)
        self.loss = nn.MSELoss()

        self.obs_keys = obs_keys
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
        obs = batch['obs']
        act = batch['actions']
        loss_val = 0
        if validate:
            policy_act = self.get_action(obs)
            loss_val = self.loss(policy_act, act).data.cpu().numpy()
        else:
            for i in range(len(self.actor)):
                obs = self.actor[i](obs)
            policy_act = obs

            loss = self.loss(policy_act, act)
            loss_val = loss.data.cpu().numpy()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss_val

    def get_action(self, obs):
        """
        Args:
        - obs (torch.Tensor): (B, 1, Obs_dim)
        Returns:
        - action (torch.Tensor): (B, 1, Action_dim)
        """
        self.eval()
        with torch.no_grad():
            for i in range(len(self.actor)):
                obs = self.actor[i](obs)
            action = obs

        return action

    def reset(self):
        return

    def is_plan_reset(self,):
        return True

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        # get epoch from end of path
        self.load_state_dict(torch.load(path))

    def set_eval(self):
        self.eval()
