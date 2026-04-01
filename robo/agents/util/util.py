import os
import sys
sys.path.append('./robomimic/')
sys.path.append('./robosuite/')
from agents.util.helper import rollout, vlm_rollout, load_data_for_training
import imageio
import robomimic.utils.obs_utils as ObsUtils
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

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


def get_folder(path_to_dlm):
    # set download folder for EASY task
    lift_folder = os.path.join(path_to_dlm, 'lift/ph')
    os.makedirs(lift_folder, exist_ok=True)
    # enforce that the dataset exists
    lift_path = os.path.join(lift_folder, "low_dim_v15.hdf5")
    assert os.path.exists(lift_path), lift_path + " does not exist"


    # set download folder for MEDIUM task
    square_folder = os.path.join(path_to_dlm, 'square/ph')
    os.makedirs(square_folder, exist_ok=True)
    # enforce that the dataset exists
    square_path = os.path.join(square_folder, "low_dim_v15.hdf5")
    assert os.path.exists(square_path), square_path + " does not exist"


    # set download folder for HARD task
    tool_hang_folder = os.path.join(path_to_dlm, 'tool_hang/ph')
    os.makedirs(tool_hang_folder, exist_ok=True)
    # enforce that the dataset exists
    tool_hang_path = os.path.join(tool_hang_folder, "low_dim_v15.hdf5")
    assert os.path.exists(tool_hang_path), tool_hang_path + " does not exist"



    input_dim_dict = {
    'square': ['square','ph', "low_dim", 23,square_folder,square_path],
    'lift': ['lift','ph', "low_dim", 19,lift_folder,lift_path],
    'tool_hang': ['tool_hang','ph', "low_dim", 53,tool_hang_folder,tool_hang_path],
                }
    return input_dim_dict


def get_dataset(download_folder,seq_len=1,batch_size=100):
    dataset_path = os.path.join(download_folder, "low_dim_v15.hdf5")
     
    train_loader, valid_loader = load_data_for_training(
        dataset_path=dataset_path,
        obs_keys=obs_keys,
        seq_len=seq_len,
        batch_size=batch_size
    )

    return dataset_path,train_loader,valid_loader

def try_rollout(model,save_path,model_file,video_path,dataset_path):
    model.load(os.path.join(save_path, model_file))

    # create a video writer
    video_file =  os.path.join(video_path,"rollout.mp4")
    video_writer = imageio.get_writer(video_file, fps=20)

    # Default rollout count for quick local checks.
    num_rollouts = 2

    success_rate = rollout(model,
                        dataset_path,
                        horizon = 400,
                        video_writer = video_writer,
                        obs_keys = obs_keys,
                        num_rollouts = num_rollouts)
    print("Success rate over {} rollouts: {}".format(num_rollouts, success_rate))

    video_writer.close()

def try_vlm_rollout(model,vlm,video_path,dataset_path,save_path=None,subtask_ind=None,instruction_path_list=[],subtask_progress_key=None,horizon=400,obs_len=1,generate_img=False,generate_vid=False,video_skip=5,img_skip=5,generate_rollout_vid=False,num_rollouts=1,figure_path="",language_path="",verbose=False):

    # Use the caller-provided rollout count.
    num_rollouts = num_rollouts

    success_rate,full_success, rollout_results = vlm_rollout(
                                            policy = model,
                                            dataset_path = dataset_path,
                                            vlm = vlm,
                                            instruction_path_list=instruction_path_list,
                                            subtask_ind=subtask_ind,
                                            subtask_progress_key=subtask_progress_key,
                                            horizon = horizon,
                                            generate_img=generate_img,
                                            generate_vid=generate_vid,
                                            video_skip=video_skip,
                                            img_skip=img_skip,
                                            generate_rollout_vid = generate_rollout_vid,
                                            obs_keys = obs_keys,
                                            obs_len=obs_len,
                                            num_rollouts = num_rollouts,
                                            figure_path=figure_path,
                                            video_path=video_path,
                                            language_path=language_path,
                                            verbose=verbose)
    print("Success rate over {} rollouts: {}".format(num_rollouts, success_rate))

    if save_path is not None:
        result_file = os.path.join(save_path,'data.pkl')
        print("Saving Results Files on ", result_file)
        with open(result_file, "wb") as f:    
            pickle.dump([full_success,rollout_results], f)



def test_rollout(model, save_path, dataset_path, model_name, obs_len=1, verbose="", num_rollouts=50):
    # model epochs saved to save_path/epoch_x.pth where x is every 50 epochs
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    success = [0]
    for name in model_name:
        print('-----------------------------')
        print("Task Name {} Model Name {} ".format(dataset_path, name))
        print("Model: ", verbose)
        print('Save Folder', save_path)
        print('-----------------------------')
        # TESTING THE MODEL WITH THE LOWEST VALIDATION ERROR
        model.load(os.path.join(save_path, name))

        # rollout
        success_rate = rollout(model,
                               dataset_path,
                               horizon=700,
                               video_writer=None,
                               obs_keys=obs_keys,
                               num_rollouts=num_rollouts,
                               obs_len=obs_len)
        success.append(success_rate)
        print("Success rate over {} rollouts: {}".format(
            num_rollouts, success_rate))

    if num_rollouts >= 50:
        data_file = os.path.join(save_path, "success.pkl")
        with open(data_file, "wb") as outfile:
            pickle.dump(success, outfile)

def plot_fig(save_path, epoch_per_iter=50):
    data_file = os.path.join(save_path, "loss.pkl")
    data_file = open(data_file, "rb")
    train_losses, valid_losses = pickle.load(data_file)
    vl = torch.tensor(valid_losses).cpu().numpy()
    tl = torch.tensor(train_losses).cpu().numpy()

    data_file = os.path.join(save_path, "success.pkl")
    data_file = open(data_file, "rb")
    success = pickle.load(data_file)
    ss = np.array(success)
    epoch = epoch_per_iter*np.arange(len(ss))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(tl[:, 1], tl[:, 0], linewidth=2, label='Training Loss')
    ax[0].plot(vl[:, 1], vl[:, 0], linewidth=2, label='Validation Loss')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training & Validation Loss (BC)")
    ax[0].legend()

    ax[1].plot(epoch, ss, linewidth=2)
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Success Rate")
    ax[1].set_title("Success Rate")
    ax[1].set_ylim([0, 1.05])

    fig.show()
    save_file = os.path.join(save_path, 'result.png')
    fig.savefig(save_file)
    pass

def plot_fig_all(path_to_dlm, save_name, args):

    save_path = os.path.join(path_to_dlm, save_name)

    fig, ax = plt.subplots(1, len(args), figsize=(6*len(args), 6))
    for ii in range(len(args)):
        task_name, num_epochs, param, title_name = args[ii]
        for d in param:
            file_path = os.path.join(save_path, task_name+"/"+str(d))

            # data_file = os.path.join(save_path, "loss.pkl")
            # data_file= open(data_file, "rb")
            # valid_losses,train_losses= pickle.load(data_file)
            success_file = os.path.join(file_path, "success.pkl")
            success_file = open(success_file, "rb")
            success = pickle.load(success_file)

            ss = np.array(success)
            epoch = np.linspace(0, num_epochs-1, len(ss))

            ax[ii].plot(epoch, ss, linewidth=2, label=str(d))
            ax[ii].set_xlabel("Epochs", fontsize=16)
            ax[ii].set_title(title_name)
            ax[ii].set_ylim([0, 1.05])
            if ii == 0:
                ax[ii].set_ylabel("Success Rate", fontsize=16)
            else:
                ax[ii].set_yticks([])
            ax[ii].legend(param)
    fig.tight_layout()
    fig.savefig(save_path + '/result.png')
    print('Saving Figure')
