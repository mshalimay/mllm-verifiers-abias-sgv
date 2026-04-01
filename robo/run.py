from time import time
from agents.util.util import get_folder, get_dataset, try_rollout, plot_fig, plot_fig_all
from agents.rl.model_util import get_model, train_mlp, train_rnn, train_transformer, train_diffusion
from agents.util.util import test_rollout
import os
import argparse

result_path = "results/datasets"
output_dim = 7


def run_mlp(yes_train=True, yes_rollout=True):
    task,dataset_type, hdf5_type, input_dim,download_folder,dataset_path = input_dim_dict['tool_hang']
    save_path = os.path.join("trainings",task,"mlp")
    if yes_train:
        model = train_mlp(result_path,download_folder,input_dim,output_dim,
                    num_epochs = 251,seq_len=1,batch_size =100,save_path = save_path)
    else:
        model = get_model(model_name='mlp',input_dim=input_dim)
    model_name = ["epoch_50.pth","epoch_100.pth","epoch_150.pth","epoch_200.pth","epoch_250.pth"]
    
    if yes_rollout:
        test_rollout(model,
                save_path= save_path,
                dataset_path=dataset_path,
                model_name=model_name,
                verbose="MLP")
    plot_fig(save_path)

def run_rnn(yes_train=True,yes_rollout=True):
    task,dataset_type, hdf5_type, input_dim,download_folder,dataset_path = input_dim_dict['tool_hang']

    hidden_dim = 400
    num_layers = 2
    seq_len = 10
    args = [input_dim,hidden_dim,num_layers,output_dim,seq_len]
    save_path = os.path.join("trainings",task,"rnn")
    if yes_train:
        model = train_rnn(result_path,download_folder,input_dim,output_dim,
               num_epochs = 501,seq_len=seq_len,batch_size =100,
               hidden_dim=hidden_dim,num_layers=num_layers,save_path = save_path)
    else:
        model = get_model(model_name='rnn',input_dim=input_dim)
    model_name = ["epoch_50.pth","epoch_100.pth","epoch_150.pth","epoch_200.pth","epoch_250.pth",
                  "epoch_300.pth","epoch_350.pth","epoch_400.pth","epoch_450.pth","epoch_500.pth"]
    if yes_rollout:
        test_rollout(model,
                save_path= save_path,
                dataset_path=dataset_path,
                model_name=model_name,
                verbose="RNN")
    plot_fig(save_path)

def run_transformer(yes_train=True, yes_rollout=True):
    task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
        'tool_hang']

    seq_len = 10  # same as prediction horizon
    denoising_steps = 100
    action_horizon = 8
    obs_horizon = 2
    args = [input_dim, denoising_steps, action_horizon,
            obs_horizon, output_dim, seq_len]
    save_path = os.path.join("trainings",task,"transformer")
    if yes_train:
        model = train_transformer(result_path, download_folder, input_dim, output_dim,
                                  num_epochs=50, seq_len=seq_len, batch_size=100,
                                  denoising_steps=denoising_steps, action_horizon=action_horizon, obs_horizon=obs_horizon,
                                  save_path=save_path)

    else:
        model = get_model(model_name='transformer', input_dim=input_dim)
    model_name = ["epoch_50.pth"]
    if yes_rollout:
        test_rollout(model,
                     save_path=save_path,
                     dataset_path=dataset_path,
                     model_name=model_name,
                     obs_len=obs_horizon,
                     verbose="Transformer",
                     num_rollouts=2)
        try_rollout(model, save_path, model_name[0], save_path, dataset_path)

def run_diffusion(yes_train=True, yes_rollout=True):
    task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
        'tool_hang']

    seq_len = 16  # same as prediction horizon
    denoising_steps = 100
    action_horizon = 8
    obs_horizon = 2
    save_path = os.path.join("trainings",task,"diffusion")
    if yes_train:
        model = train_diffusion(result_path, download_folder, input_dim, output_dim,
                                num_epochs=101, seq_len=seq_len, batch_size=100,
                                denoising_steps=denoising_steps, action_horizon=action_horizon, obs_horizon=obs_horizon,
                                save_path=save_path)

    else:
        model = get_model(model_name='diffusion', input_dim=input_dim)
    model_name = ["epoch_100.pth"]
    if yes_rollout:
        test_rollout(model,
                     save_path=os.path.join(result_path, save_path),
                     dataset_path=dataset_path,
                     model_name=model_name,
                     obs_len=obs_horizon,
                     verbose="Diffusion",
                     num_rollouts=50)
        try_rollout(model, os.path.join(result_path, save_path),
                    model_name[0], os.path.join(result_path, save_path), dataset_path)
    plot_fig(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_train', dest='yes_train', action='store_false')
    parser.add_argument('--no_rollout', dest='yes_rollout', action='store_false')
    parser.add_argument('-m','--model_name', type=str, default='mlp', choices=['mlp', 'rnn', 'transformer', 'diffusion'])
    args = parser.parse_args()
    yes_train = args.yes_train
    yes_rollout = args.yes_rollout
    model_name = args.model_name

    input_dim_dict = get_folder(result_path)

    if model_name == 'mlp':
        run_mlp(yes_train=yes_train, yes_rollout=yes_rollout)
    elif model_name == 'rnn':
        run_rnn(yes_train=yes_train, yes_rollout=yes_rollout)
    elif model_name == 'transformer':
        run_transformer(yes_train=yes_train, yes_rollout=yes_rollout)
    elif model_name == 'diffusion':
        run_diffusion(yes_train=yes_train, yes_rollout=yes_rollout)

    

