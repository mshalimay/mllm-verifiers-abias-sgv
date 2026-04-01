import sys
sys.path.append('./robomimic/')
sys.path.append('./robosuite/')

import os
from agents.util import helper
from agents.util.helper import rollout
import imageio
from agents.rl.model.mlp import DLMMLP
from agents.rl.model.rnn import DLM_RNN
from agents.rl.model.diffusion import DLM_Diffusion
from agents.rl.model.transformer import DLM_Transformer
import robomimic.utils.obs_utils as ObsUtils
from agents.util.util import get_dataset
from agents.util.helper import train
import pickle

obs_spec = dict(
    obs=dict(
            low_dim=[
                    "object",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    ],
            rgb=[],
        ),
)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]   

obs_spec = dict(
    obs=dict(
            low_dim=[
                    "object",
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    ],
            rgb=[],
        ),
)
ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
obs_keys = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]   

def get_model(model_name='mlp',input_dim=None):
    if 'mlp' in model_name:
        model = DLMMLP(input_dim=input_dim, hidden_dims=[1024]*5, output_dim=7, obs_keys = obs_keys)
    elif 'rnn' in model_name:
        model = DLM_RNN(input_dim = input_dim, hidden_dim = 400,
                        num_layers = 2, output_dim = 7,
                        obs_keys = obs_keys, rnn_horizon = 8)
    elif 'diffusion' in model_name:
        model = DLM_Diffusion(
            output_dim = 7,
            input_dim = input_dim,
            denoising_steps=100,
            prediction_horizon=16,
            action_horizon=8,
            obs_horizon = 2
            )
    elif 'transformer' in model_name:
        model = DLM_Transformer(
            output_dim=7,
            input_dim=input_dim,
            denoising_steps=100,
            prediction_horizon=16,
            action_horizon=8,
            obs_horizon=2
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model



def train_mlp(path_to_dlm,download_folder,input_dim,output_dim,
               num_epochs = 251,seq_len=1,batch_size =100,
               hidden_dims = [1024]*5,save_path = "trainings/mlp"):
    _,train_loader,valid_loader = get_dataset(download_folder,seq_len,batch_size)
    
    model = DLMMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, obs_keys = obs_keys)
    
    # model epochs saved to save_path/epoch_x.pth where x is every 50 epochs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train_losses and valid_losses are lists of (loss, epoch) tuples
    train_losses, valid_losses = train(model, train_loader, valid_loader, num_epochs=num_epochs, save_path=save_path)


    data_file = os.path.join(save_path, "loss.pkl")
    with open(data_file, "wb") as outfile:
        pickle.dump([valid_losses,train_losses], outfile)

    return model

def train_rnn(path_to_dlm,dataset_path,input_dim,output_dim,
               num_epochs = 501,seq_len=10,batch_size =100,
               hidden_dim=400,num_layers=2,save_path = "trainings/rnn"):
    

    _,train_loader,valid_loader = get_dataset(dataset_path,seq_len,batch_size)
    model = DLM_RNN(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim, obs_keys = obs_keys, rnn_horizon = seq_len)
    
    # model epochs saved to save_path/epoch_x.pth where x is every 50 epochs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train_losses and valid_losses are lists of (loss, epoch) tuples
    train_losses, valid_losses = train(model, train_loader, valid_loader, num_epochs=num_epochs, save_path=save_path)


    data_file = os.path.join(save_path, "loss.pkl")
    with open(data_file, "wb") as outfile:
        pickle.dump([valid_losses,train_losses], outfile)
    
    return model

def train_transformer(path_to_dlm, dataset_path, input_dim, output_dim,
                      num_epochs=51, seq_len=16, batch_size=100,
                      denoising_steps=100, action_horizon=8, obs_horizon=2,
                      save_path="trainings/transformer"):

    _, train_loader, valid_loader = get_dataset(
        dataset_path, seq_len, batch_size)

    model = DLM_Transformer(
        output_dim=output_dim,
        input_dim=input_dim,
        denoising_steps=denoising_steps,
        prediction_horizon=seq_len,
        action_horizon=action_horizon,
        obs_horizon=obs_horizon
    )

    # model epochs saved to save_path/epoch_x.pth where x is every 50 epochs
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(save_path)

    controller_name = os.listdir(save_path)
    cp = 0
    # if 'epoch_0.pth' in controller_name:
    #     controller_name = [i for i in controller_name if '0.pth' in i]
    #     controller_id = controller_id = sorted(
    #         [int(i.split('epoch_')[1].split('.')[0]) for i in controller_name])
    #     cp = controller_id[-1]
    #     cp_path = os.path.join(save_path, f'epoch_{cp}.pth')
    #     print("Loading Checkpoint: ", cp_path)
    #     model.load(cp_path)
    #     model.epoch = cp

    # train_losses and valid_losses are lists of (loss, epoch) tuples
    train_losses, valid_losses = train(
        model, train_loader, valid_loader, start_epoch=cp, num_epochs=num_epochs-cp, save_path=save_path)

    data_file = os.path.join(save_path, "loss.pkl")
    with open(data_file, "wb") as outfile:
        pickle.dump([valid_losses, train_losses], outfile)

    return model

def train_diffusion(path_to_dlm, dataset_path, input_dim, output_dim,
                    num_epochs=51, seq_len=16, batch_size=100,
                    denoising_steps=100, action_horizon=8, obs_horizon=2,
                    save_path="trainings/diffusion"):

    _, train_loader, valid_loader = get_dataset(
        dataset_path, seq_len, batch_size)

    model = DLM_Diffusion(
        output_dim=output_dim,
        input_dim=input_dim,
        denoising_steps=denoising_steps,
        prediction_horizon=seq_len,
        action_horizon=action_horizon,
        obs_horizon=obs_horizon
    )

    # model epochs saved to save_path/epoch_x.pth where x is every 50 epochs
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    controller_name = os.listdir(save_path)
    cp = 0
    if 'epoch_0.pth' in controller_name:
        controller_name = [i for i in controller_name if '0.pth' in i]
        controller_id = controller_id = sorted(
            [int(i.split('epoch_')[1].split('.')[0]) for i in controller_name])
        cp = controller_id[-1]
        cp_path = os.path.join(save_path, f'epoch_{cp}.pth')
        print("Loading Checkpoint: ", cp_path)
        model.load(cp_path)
        model.epoch = cp

    # train_losses and valid_losses are lists of (loss, epoch) tuples
    train_losses, valid_losses = train(
        model, train_loader, valid_loader, start_epoch=cp, num_epochs=num_epochs-cp, save_path=save_path)

    data_file = os.path.join(save_path, "loss.pkl")
    with open(data_file, "wb") as outfile:
        pickle.dump([valid_losses, train_losses], outfile)

    return model