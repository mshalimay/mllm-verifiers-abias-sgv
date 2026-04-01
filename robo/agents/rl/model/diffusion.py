import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from diffusers import DDPMScheduler
from agents.util.helperModels import ConditionalUnet1D
from diffusers.training_utils import EMAModel


class DLM_Diffusion(nn.Module):
    def __init__(
        self,
        output_dim,
        input_dim,
        denoising_steps,
        obs_horizon,
        prediction_horizon,
        action_horizon,
    ):

        super(DLM_Diffusion, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.denoising_steps = denoising_steps
        self.output_dim = output_dim
        self.obs_horizon = obs_horizon
        self.prediction_horizon = prediction_horizon
        self.action_horizon = action_horizon
        self.input_dim = input_dim

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=output_dim,
            global_cond_dim=int(input_dim*obs_horizon),
        )

        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.denoising_steps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.noise_pred_net.to(self.device)
        self.optimizer = optim.Adam(
            self.noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
        self.ema = EMAModel(
            parameters=self.noise_pred_net.parameters(), power=0.75)
        self.ema.to(self.device)
        self.epoch = 0

    def train_on_batch(self, batch, validate):
        """
        Args:
        - batch (dict): batch of data batch['obs'].shape = (B, T, Obs_dim) and batch['actions'].shape = (B, T, Action_dim)
        - validate (bool): whether batch is for validation
        Returns:
        - loss_val (float): value of loss
        """
        loss_val = None
        self.noise_pred_net.train()
        obs = batch['obs']
        B = obs.shape[0]
        act = batch['actions']

        # Get the noise and cond
        t = torch.randint(0, self.denoising_steps, (B,)).to(self.device).long()
        noise = torch.randn(act.shape).to(self.device)
        cond = torch.flatten(obs[:, :self.obs_horizon, :], 1)
        if validate:
            with torch.no_grad():
                # Add noise to action
                sample = self.scheduler.add_noise(act, noise, t)
                pred = self.noise_pred_net(sample, t, cond).to(self.device)
                loss = torch.nn.MSELoss()(noise, pred)
        else:
            # Add noise to action
            sample = self.scheduler.add_noise(act, noise, t)
            pred = self.noise_pred_net(sample, t, cond).to(self.device)
            loss = torch.nn.MSELoss()(noise, pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.step(self.noise_pred_net.parameters())

        loss_val = loss.item()

        return loss_val

    def get_action(self, obs):
        """
        Args:
        - obs (dict): dictionary of observations of shape (B, prediction_horizon, Obs_dim)
        Returns:
        - actions (np.ndarray): action of shape (B, action_horizon, Action_dim)
        """
        original_params = deepcopy(self.noise_pred_net.state_dict())
        self.ema.copy_to(self.noise_pred_net.parameters())
        self.noise_pred_net.eval()

        self.scheduler.set_timesteps(self.denoising_steps)
        B = obs.shape[0]
        obs = torch.flatten(obs, 1)

        with torch.no_grad():
            input_shape = (B, self.prediction_horizon, self.output_dim)
            noise = torch.randn(input_shape).to(self.device)
            cond = obs

            for t in self.scheduler.timesteps:
                pred = self.noise_pred_net(noise, t, cond)
                traj = self.scheduler.step(pred, t, noise)
                noise = traj.prev_sample

            start = self.obs_horizon-1
            end = start + self.action_horizon
            actions = noise[:, start:end]

        # restore original params
        self.noise_pred_net.load_state_dict(original_params)

        return actions

    def reset(self):
        return

    def is_plan_reset(self,):
        return True

    def save(self, path):
        self.ema.copy_to(self.noise_pred_net.parameters())
        torch.save(self.state_dict(), path)
        # save the ema model, put ema before the .pth
        torch.save(self.ema.state_dict(), path.replace(".pth", "ema.pth"))

    def load(self, path):
        # get epoch from end of path
        self.load_state_dict(torch.load(path))
        self.ema.load_state_dict(torch.load(path.replace(".pth", "ema.pth")))

    def set_horizon(self, horizon):
        self.prediction_horizon = horizon
        self.action_horizon = horizon

    def set_eval(self):
        self.eval()
