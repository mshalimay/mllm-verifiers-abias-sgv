import unittest
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, module="OpenGL.platform.egl")

sys.path.append('./robomimic/')
sys.path.append('./robosuite/')


from pathlib import Path
import h5py
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY, HF_REPO_ID
from agents.util.util import get_folder, try_vlm_rollout
from agents.rl.model_util import get_model
from agents.rl.model import mlp, rnn, diffusion
from agents.vlm.get_vlm_model import get_vlm_model

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

    #Each observation is a dictionary that maps modalities to numpy arrays of shape (time, obs modality dim)
    print("observations:")
    for obs, obs_key in demo_grp["obs"].items():
        print("{} - shape {}".format(obs , obs_key.shape))


class TestCaseBase(unittest.TestCase):

    def setUp(self):
        file_path = Path(__file__).parent
        self.vid_path = file_path / "video"  #Store the video for saving
        self.fix_path = file_path.parent / "models"
        self.figure_path = file_path / "figure"  #Store the fix model path
        self.language_path = file_path / "language"  #Store the fix model path
        self.download_path = 'results/datasets'  #Store the data
        self.result_path = file_path / "results" 
        self.instruction_step_path = file_path / "instruction"  #Store the fix model path
        self.vlm_path = 'vlm'  #VLM data
        self.instruction_path = 'instruction'


        os.makedirs(self.vid_path, exist_ok=True)
        os.makedirs(self.fix_path, exist_ok=True)
        os.makedirs(self.figure_path, exist_ok=True)
        os.makedirs(self.download_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)
        self.task_list = ['tool_hang']
        self.task_list_subtask_ind = {'tool_hang': [42, 43]}
        self.task_list_subtask_progress_key = {'tool_hang': ['is inserted','is hanging']}
        self.model_list = ['diffusion']
        self.vlm_list = ['gpt-4o']

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

        assert os.path.exists(dataset_path)
        print('\nTask Description: ', task)
        print_obs(dataset_path)

    def assertModelLoad(self, model_name, input_dim, save_path):
        model = get_model(model_name, input_dim)
        model.load(os.path.join(save_path, "model.pth"))
        return model



class TestRobomimic(TestCaseBase):
    """Tests for the RoboMimic"""

    def test_image_gen(self):
        print("-------------------------------- TEST CASE: Generating images --------------------------------")
        input_dim_dict = get_folder(self.download_path)
        horizon = 10
        model_horizon = 0
        num_rollouts = 2
        generate_img = True
        for model_name in self.model_list:
            print("Model to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]
                subtask_ind = self.task_list_subtask_ind[task]
                subtask_progress_key = self.task_list_subtask_progress_key[task]

                #Get the model save path
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)
                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                model.load(os.path.join(save_path, "model.pth"))

                with self.subTest("Testing Image Generation"):
                    video_path = os.path.join(self.vid_path, task, 'ph',
                                              model_name)
                    figure_path = os.path.join(self.figure_path, task, 'ph',
                                               model_name)
                    result_path = os.path.join(self.result_path, task, 'ph',
                                               model_name)
                    language_path = os.path.join(self.language_path, task, 'ph',
                                               model_name)
                    os.makedirs(figure_path, exist_ok=True)
                    os.makedirs(language_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(video_path, exist_ok=True)
                    try_vlm_rollout(model=model,
                                    vlm=None,
                                    dataset_path=dataset_path,
                                    save_path=result_path,
                                    video_path=video_path,
                                    subtask_ind=subtask_ind,
                                    generate_img=generate_img,
                                    horizon=horizon,
                                    num_rollouts=num_rollouts,
                                    figure_path=figure_path,
                                    language_path=language_path,
                                    subtask_progress_key=subtask_progress_key)

    def test_vid_gen(self):
        print("-------------------------------- TEST CASE: Generating videos --------------------------------")
        input_dim_dict = get_folder(self.download_path)
        horizon = 50
        model_horizon = 0
        num_rollouts = 2
        generate_vid = True
        for model_name in self.model_list:
            print("Model to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]
                subtask_ind = self.task_list_subtask_ind[task]
                subtask_progress_key = self.task_list_subtask_progress_key[task]

                #Get the model save path
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)
                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                model.load(os.path.join(save_path, "model.pth"))

                with self.subTest("Testing Videos Generation"):
                    video_path = os.path.join(self.vid_path, task, 'ph',
                                              model_name)
                    figure_path = os.path.join(self.figure_path, task, 'ph',
                                               model_name)
                    result_path = os.path.join(self.result_path, task, 'ph',
                                               model_name)
                    language_path = os.path.join(self.language_path, task, 'ph',
                                               model_name)
                    os.makedirs(figure_path, exist_ok=True)
                    os.makedirs(language_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(video_path, exist_ok=True)
                    try_vlm_rollout(model = model,
                                    vlm = None,
                                    dataset_path = dataset_path,
                                    video_path = video_path,
                                    subtask_ind=subtask_ind,
                                    generate_vid=generate_vid,
                                    horizon=horizon,
                                    num_rollouts=num_rollouts,
                                    figure_path=figure_path,
                                    language_path=language_path,
                                    save_path=result_path,
                                    subtask_progress_key=subtask_progress_key)

    def test_data(self):
        print("-------------------------------- TEST CASE: Dataset Quality --------------------------------")
        input_dim_dict = get_folder(self.download_path)
        horizon = 10
        model_horizon = 0
        generate_img=False
        for model_name in self.model_list:
            print("Model to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]
                subtask_ind = self.task_list_subtask_ind[task]
                subtask_progress_key = self.task_list_subtask_progress_key[task]

                #Get the model save path
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)
                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                    if model_horizon > 0:
                        model.set_horizon(model_horizon)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)
                    if model_horizon > 0:
                        model.set_horizon(model_horizon)
                model.load(os.path.join(save_path, "model.pth"))

                with self.subTest("Testing Data Quality"):
                    video_path = os.path.join(self.vid_path, task, 'ph',
                                              model_name)
                    figure_path = os.path.join(self.figure_path, task, 'ph',
                                               model_name)
                    result_path = os.path.join(self.result_path, task, 'ph',
                                               model_name)
                    language_path = os.path.join(self.language_path, task, 'ph',
                                               model_name)
                    os.makedirs(figure_path, exist_ok=True)
                    os.makedirs(language_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(video_path, exist_ok=True)
                    try_vlm_rollout(model = model,
                                    vlm = None,
                                    video_path = video_path,
                                    dataset_path = dataset_path,
                                    save_path = result_path,
                                    subtask_ind=subtask_ind,
                                    generate_img=generate_img,
                                    horizon=horizon,
                                    num_rollouts=2,
                                    figure_path=figure_path,
                                    language_path=language_path,
                                    subtask_progress_key=subtask_progress_key)
                    
                    result_file = os.path.join(result_path,'data.pkl')
                    with open(result_file, "rb") as f:
                        result_data = pickle.load(f)
                    results = result_data[1]
                    replan_data = results[0]['replan']
                    self.assertEqual(len(replan_data),horizon)

                    subtask_data = results[0]['subtask']
                    self.assertEqual(len(subtask_data),horizon)

    def test_no_vlm(self):
        input_dim_dict = get_folder(self.download_path)
        horizon = 400
        model_horizon = 0
        num_rollouts = 2
        generate_img=True
        generate_vid=False
        vlm = None
        for model_name in self.model_list:
            print("Algorithm to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]
                
                #Get the subtask index of the observation for a specific task
                subtask_ind = self.task_list_subtask_ind[task]

                #Get the model save path
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)
                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                model.load(os.path.join(save_path, "model.pth"))

                with self.subTest("Testing Without any VLM"):
                    video_path = os.path.join(self.vid_path, task, 'ph',
                                              model_name)
                    figure_path = os.path.join(self.figure_path, task, 'ph',
                                               model_name)
                    result_path = os.path.join(self.result_path, task, 'ph',
                                               model_name)
                    language_path = os.path.join(self.language_path, task, 'ph',
                                               model_name)
                    os.makedirs(figure_path, exist_ok=True)
                    os.makedirs(language_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    os.makedirs(video_path, exist_ok=True)
                    try_vlm_rollout(model=model,
                                    vlm=vlm,
                                    video_path=video_path,
                                    dataset_path=dataset_path,
                                    save_path=result_path,
                                    subtask_ind=subtask_ind,
                                    generate_img=generate_img,
                                    generate_vid=generate_vid,
                                    horizon=horizon,
                                    num_rollouts=num_rollouts,
                                    figure_path=figure_path,
                                    language_path=language_path,
                                    )

    def test_vlm(self):
        print("-------------------------------- TEST CASE: VLM --------------------------------")
        input_dim_dict = get_folder(self.download_path)
        horizon = 10
        model_horizon = horizon
        num_rollouts = 2
        img_skip=horizon//2
        generate_img=True

        vlm_name = self.vlm_list[0]
        vlm = get_vlm_model(vlm_name, temperature_set=0, max_tokens=3000)
        api_key = vlm.check_api_key()
        assert api_key is not None, "OPENAI_API_KEY is not set"
        
        print("VLM to be used: ", vlm_name)
        for model_name in self.model_list:
            print("Model to be used: ", model_name)
            for task in self.task_list:
                task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
                    task]

                #Get the subtask index of the observation for a specific task
                subtask_ind = self.task_list_subtask_ind[task]
                subtask_progress_key = self.task_list_subtask_progress_key[task]
                #Get the instructions
                instruction_path_list = [os.path.join(self.instruction_path,task+'.txt'), os.path.join(self.instruction_path,task+'_1pass_prompt.txt')]

                #Get the model save path
                save_path = os.path.join(self.fix_path, task, 'ph', model_name)
                model = self.assertModelLoad(model_name, input_dim, save_path)
                if model_name == 'mlp':
                    self.assertIsInstance(model, mlp.DLMMLP)
                elif model_name == 'rnn':
                    self.assertIsInstance(model, rnn.DLM_RNN)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                        self.assertEqual(horizon, model.hidden_state_horizon)
                elif model_name == 'diffusion':
                    self.assertIsInstance(model, diffusion.DLM_Diffusion)
                    if model_horizon > 0:
                        model.set_horizon(horizon)
                        self.assertEqual(horizon, model.action_horizon)
                model.load(os.path.join(save_path, "model.pth"))
                

                with self.subTest("Testing VLMs"):
                    video_path = os.path.join(self.vid_path, task, 'ph',
                                              model_name+"_"+vlm_name)
                    figure_path = os.path.join(self.figure_path, task, 'ph',
                                               model_name+"_"+vlm_name)
                    language_path = os.path.join(self.language_path, task, 'ph',
                                               model_name+"_"+vlm_name)
                    result_path = os.path.join(self.result_path, task, 'ph',
                                               model_name+"_"+vlm_name)
                    os.makedirs(video_path, exist_ok=True)
                    os.makedirs(figure_path, exist_ok=True)
                    os.makedirs(language_path, exist_ok=True)
                    os.makedirs(result_path, exist_ok=True)
                    try_vlm_rollout(model=model,
                                    vlm=vlm,
                                    video_path=video_path,
                                    dataset_path=dataset_path,
                                    save_path=result_path,
                                    subtask_ind=subtask_ind,
                                    instruction_path_list = instruction_path_list,
                                    subtask_progress_key=subtask_progress_key,
                                    horizon=horizon,
                                    generate_img=generate_img,
                                    img_skip=img_skip,
                                    num_rollouts=num_rollouts,
                                    figure_path=figure_path,
                                    language_path=language_path)


if __name__ == '__main__':
    unittest.main()
