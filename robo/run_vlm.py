import warnings
warnings.filterwarnings("ignore", category=ResourceWarning, module="OpenGL.platform.egl")

from time import time
from agents.util.util import *
from agents.util.helper import generate_vlm_demos
import os
import sys
from agents.util.util import get_folder, try_vlm_rollout
from agents.rl.model_util import get_model
from agents.vlm.get_vlm_model import get_vlm_model
import argparse


output_dim = 7
result_path = "results"
download_path = os.path.join(result_path, 'datasets')
instruction_path = os.path.join('instruction')
fixture_path = os.path.join('models')


os.makedirs(result_path, exist_ok=True)
os.makedirs(download_path, exist_ok=True)
os.makedirs(fixture_path, exist_ok=True)


def ModelLoad(model_name, input_dim, save_path):
    model = get_model(model_name, input_dim)
    model.load(os.path.join(save_path, "model.pth"))
    return model


def make_videos():
    horizon = 400
    model_horizon = 0
    num_rollouts = -1
    generate_img = True
    generate_vid = False
    vlm = None
    task = 'tool_hang'
    dataset_path = os.path.join(download_path, 'tool_hang', 'ph', 'low_dim_v15.hdf5')

    figure_path = os.path.join('results', 'figure', task, 'ph',
                               'expert')
    os.makedirs(figure_path, exist_ok=True)
    generate_vlm_demos(figure_path, dataset_path, num_rollouts)


def run_rollout(vlm_name=None, model_name='rnn', num_rollouts=1, horizon=700, model_horizon=0, generate_img=False, generate_vid=False, generate_rollout_vid=False, img_skip=5, video_skip=5, verbose=False):
    task = 'tool_hang'
    subtask_ind = [42, 43]
    subtask_progress_key = ['is inserted', 'is hanging']
    if vlm_name is None:
        vlm_name = 'noVLM'
        vlm = None
    else:
        vlm = get_vlm_model(vlm_name, temperature_set=0, max_tokens=3000)

    # Get the needed data from the task
    task, dataset_type, hdf5_type, input_dim, download_folder, dataset_path = input_dim_dict[
        task]

    instruction_path_list = [
        "instruction/tool_hang.txt"
    ]
    if vlm is not None:
        if '1pass' in vlm_name:
            instruction_path_list = [
                "instruction/tool_hang.txt",
                "instruction/tool_hang_1pass_prompt.txt",
            ]
        else:
            instruction_path_list = [
                "instruction/tool_hang.txt",
                "instruction/tool_hang_2pass_prompt_part1.txt",
                "instruction/tool_hang_2pass_prompt_part2.txt",
            ]

    # Get the model save path
    save_path = os.path.join(fixture_path, task, 'ph', model_name)
    model = ModelLoad(model_name, input_dim, save_path)
    obs_len = 1
    if model_name == 'mlp':
        pass
    elif model_name == 'rnn':
        if model_horizon > 0:
            model.set_horizon(model_horizon)
    elif model_name == 'diffusion':
        if model_horizon > 0:
            model.set_horizon(model_horizon)
        obs_len = model.obs_horizon

    task_path = os.path.join(result_path, task)
    model_path = os.path.join(task_path, model_name,
                              vlm_name, 'horizon_'+str(model_horizon))
    figure_path = os.path.join(model_path, 'figure')
    video_path = os.path.join(model_path, 'video')
    language_path = os.path.join(model_path, 'language')

    os.makedirs(task_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(language_path, exist_ok=True)
    try_vlm_rollout(model,
                    vlm,
                    video_path,
                    dataset_path,
                    save_path=model_path,
                    subtask_ind=subtask_ind,
                    subtask_progress_key=subtask_progress_key,
                    instruction_path_list=instruction_path_list,
                    generate_img=generate_img,
                    generate_vid=generate_vid,
                    video_skip=video_skip,
                    img_skip=img_skip,
                    generate_rollout_vid=generate_rollout_vid,
                    horizon=horizon,
                    obs_len=obs_len,
                    num_rollouts=num_rollouts,
                    figure_path=figure_path,
                    language_path=language_path,
                    verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm_name', type=str, default=None)
    parser.add_argument('-m', '--model_name', type=str, default='mlp', choices=['mlp', 'rnn', 'diffusion'])
    parser.add_argument('-n', '--num_rollouts', type=int, default=50)
    parser.add_argument('-lh', '--long_horizon', action='store_true')
    parser.add_argument('--horizon', type=int, default=700)
    args = parser.parse_args()
    num_rollouts = args.num_rollouts
    horizon = args.horizon
    long_horizon = args.long_horizon

    input_dim_dict = get_folder(download_path)

    if args.model_name == 'mlp':
        run_rollout(vlm_name=None, model_name='mlp', num_rollouts=num_rollouts, horizon=horizon, model_horizon=0)
    elif args.model_name == 'rnn':
        if long_horizon:
            run_rollout(vlm_name=None, model_name='rnn', num_rollouts=num_rollouts, horizon=horizon, model_horizon=700)
        else:
            if args.vlm_name is None:
                run_rollout(vlm_name=None, model_name='rnn',
                            num_rollouts=num_rollouts, horizon=horizon, model_horizon=0)
            else:
                run_rollout(vlm_name=args.vlm_name, model_name='rnn', num_rollouts=num_rollouts, horizon=horizon,
                            model_horizon=0, generate_img=True, img_skip=20, generate_rollout_vid=True, verbose=True)
    elif args.model_name == 'diffusion':
        if long_horizon:
            run_rollout(vlm_name=None, model_name='diffusion', num_rollouts=num_rollouts,
                        horizon=horizon, model_horizon=700, generate_rollout_vid=True, verbose=True)
        else:
            if args.vlm_name is None:
                run_rollout(vlm_name=None, model_name='diffusion', num_rollouts=num_rollouts,
                            horizon=horizon, model_horizon=0, generate_rollout_vid=True, verbose=True)
            else:
                run_rollout(vlm_name=args.vlm_name, model_name='diffusion', num_rollouts=num_rollouts,
                            horizon=horizon, model_horizon=0, generate_img=True, img_skip=20, generate_rollout_vid=True, verbose=True)
    else:
        make_videos()
