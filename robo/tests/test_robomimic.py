import warnings
import sys
warnings.filterwarnings("ignore", category=ResourceWarning, module="OpenGL.platform.egl")
sys.path.append('./robomimic/')
sys.path.append('./robosuite/')
from agents.rl.model import mlp, rnn, diffusion
from agents.rl.model_util import get_model
from agents.util.util import get_folder, get_dataset, try_rollout
from agents.util import helper
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY, HF_REPO_ID
import robomimic.utils.file_utils as FileUtils
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import unittest
import os

def print_obs(dataset_path):
    # open file
    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data".  each demonstration is named "demo_#" where # is a number, starting from 0
    demos = list(f["data"].keys())
    num_demos = len(demos)

    print("hdf5 file {} has {} demonstrations".format(dataset_path, num_demos))

    # look at first demonstration
    demo_key = demos[0]
    demo_grp = f["data/{}".format(demo_key)]

    # actions is a num numpy array of shape (time, action dim)
    actions = demo_grp["actions"][:]
    print("shape of actions {}".format(actions.shape))

    # Each observation is a dictionary that maps modalities to numpy arrays of shape (time, obs modality dim)
    print("observations:")
    for obs, obs_key in demo_grp["obs"].items():
        print("{} - shape {}".format(obs, obs_key.shape))


class TestCaseBase(unittest.TestCase):

    def setUp(self):
        file_path = Path(__file__).parent
        self.vid_path = file_path / "video"  # Store the video for saving
        self.fix_path = file_path.parent / "models"
        self.download_path = 'results/datasets'  # Store the data

        os.makedirs(self.fix_path, exist_ok=True)
        # self.task_list = ['lift', 'square', 'tool_hang']
        # self.model_list = ['mlp', 'rnn', 'diffusion']
        self.model_list = ['diffusion']
        self.task_list = ['tool_hang']

    def assertDownload(self, task="lift"):
        # set download folder
        download_folder = os.path.join(self.download_path, task, 'ph')
        os.makedirs(download_folder, exist_ok=True)

        # download the dataset (registry stores path for HF, not full URL)
        dataset_filename = "low_dim_v15.hdf5"
        dataset_path = os.path.join(download_folder, dataset_filename)
        dataset_type = "ph"
        hdf5_type = "low_dim"
        if not os.path.exists(dataset_path):
            hf_path = DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"]
            FileUtils.download_file_from_hf(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                download_dir=download_folder,
                check_overwrite=False,
            )
        # enforce that the dataset exists

        assert os.path.exists(dataset_path), f"Dataset {dataset_path} does not exist"
        print('\nTask Description: ', task)
        print_obs(dataset_path)

    def assertModelLoad(self, model_name, input_dim, save_path):
        model = get_model(model_name, input_dim)
        model.load(os.path.join(save_path, "model.pth"))
        return model


class TestRobomimic(TestCaseBase):
    """Tests for the RoboMimic"""

    def test_download(self):
        print("-------------------------------- TEST CASE: Downloading datasets --------------------------------")
        for task in self.task_list:
            self.assertDownload(task)

    def test_vid(self):
        print("-------------------------------- TEST CASE: Generating videos --------------------------------")
        for task in self.task_list:
            video_folder = os.path.join(self.vid_path, task, 'ph')
            os.makedirs(video_folder, exist_ok=True)
            dataset_path = os.path.join(
                self.download_path, task, 'ph', "low_dim_v15.hdf5")

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
            ObsUtils.initialize_obs_utils_with_obs_specs(
                obs_modality_specs=obs_spec)
            # prepare to write playback trajectories to video
            video_path = os.path.join(video_folder, "playback.mp4")
            helper.playback_demos(video_path, dataset_path, num_rollouts=2)

    def test_model_load(self):
        print("-------------------------------- TEST CASE: Loading models --------------------------------")
        input_dim_dict = get_folder(self.download_path)

        for model_name in self.model_list:
            print("Model to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)

                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)

                with self.subTest("Testing Rollouts"):
                    video_path = os.path.join(
                        self.vid_path, task, 'ph', model_name)
                    os.makedirs(video_path, exist_ok=True)
                    try_rollout(model, save_path, "model.pth",
                                video_path, dataset_path)


if __name__ == '__main__':
    unittest.main()
