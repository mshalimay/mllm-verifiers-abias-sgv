import sys
sys.path.append('./robosuite/')
sys.path.append('./robomimic/')

import imageio
import h5py
import torch
import numpy as np
from copy import deepcopy
import os
from collections import deque
from torch.utils.data import DataLoader
from robomimic.utils.dataset import SequenceDataset
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from tqdm import tqdm
from agents.vlm.vlm_helper import read_text_file
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def playback_trajectory(env, video_writer, demo_key, f):
    """
    Simple helper function to playback the trajectory stored under the hdf5 group @demo_key and
    write frames rendered from the simulation to the active @video_writer.

    Args:
        env: simulation environment instance
        video_writer: imageio video writer for recording frames
        demo_key: key identifying the demonstration in the hdf5 file
        f: open hdf5 file handle containing the demonstration data
    """

    # robosuite datasets store the ground-truth simulator states under the "states" key.
    # We will use the first one, alone with the model xml, to reset the environment to
    # the initial configuration before playing back actions.
    init_state = f["data/{}/states".format(demo_key)][0]
    model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
    initial_state_dict = dict(states=init_state, model=model_xml)

    # reset to initial state
    env.reset_to(initial_state_dict)

    # playback actions one by one, and render frames
    actions = f["data/{}/actions".format(demo_key)][:]
    for t in range(actions.shape[0]):
        env.step(actions[t])
        video_img = env.render(mode="rgb_array", height=512,
                               width=512, camera_name="agentview")
        video_writer.append_data(video_img)

def playback_demos(video_path, dataset_path, num_rollouts):
    video_writer = imageio.get_writer(video_path, fps=20)
    # create simulation environment from environment metedata=
    env = EnvUtils.create_env_from_metadata(
        env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path),
        render=False,            # no on-screen rendering
        render_offscreen=True,   # off-screen rendering to support rendering video frames
    )

    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data".  each demonstration is named "demo_#" where # is a number, starting from 0
    demos = list(f["data"].keys())

    # playback the first 5 demos
    for demo_key in demos[8:9]:
        print("Playing back demo key: {}".format(demo_key))
        init_state = f["data/{}/states".format(demo_key)][0]
        model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)
        # reset to initial state
        env.reset_to(initial_state_dict)

        # playback actions one by one, and render frames
        actions = f["data/{}/actions".format(demo_key)][:]
        for t in range(actions.shape[0]):
            env.step(actions[t])
            video_img = env.render(
                mode="rgb_array", height=512, width=512, camera_name="agentview")
            video_writer.append_data(video_img)

    # done writing video
    video_writer.close()

def generate_vlm_demos(figure_path, dataset_path, num_rollouts):

    env = EnvUtils.create_env_from_metadata(
        env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path),
        render=False,            # no on-screen rendering
        render_offscreen=True,   # off-screen rendering to support rendering video frames
    )

    f = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data".  each demonstration is named "demo_#" where # is a number, starting from 0
    demos = list(f["data"].keys())

    sus = []
    # sus = [6, 8, 9, 11, 13, 17, 20, 21, 25, 29, 31, 34, 35, 37, 48,
    #        49, 53, 54, 55, 56, 68, 69, 71, 73, 76, 79, 83, 87, 90,
    #        91, 95, 97, 98, 102, 104, 105, 110, 114, 125, 130, 133,
    #        138, 139, 143, 144, 145, 146, 148, 152, 156, 160, 163,
    #        167, 170, 176, 177, 179, 181, 182, 187, 190, 194, 196]
    # demo_keys = [
    #     103,105,106,108,11,113,116,117,120,124,126,129,131,141,
    #     142,146,147,148,149,16,160,162,164,167,17,173,177,18,
    #     180,184,186,190,192,193,198,21,31,36,
    #     39,43,44,48,49,5,50,52,56,6,63,66,
    #     7,72,78,79,80,83,88,90,94,96,
    # ]

    # playback the first 5 demos
    for i, demo_key in enumerate(tqdm(demos[:num_rollouts])):
        if i in sus:
            continue
        rollout_fig_path = os.path.join(figure_path,
                                        'rollout_{:03}'.format(i))
        os.makedirs(rollout_fig_path, exist_ok=True)

        print("Playing back demo key: {}".format(demo_key))
        init_state = f["data/{}/states".format(demo_key)][0]
        model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)
        # reset to initial state
        env.reset_to(initial_state_dict)

        # playback actions one by one, and render frames
        actions = f["data/{}/actions".format(demo_key)][:]
        for t in range(actions.shape[0]):
            env.step(actions[t])
            img = env.render(
                mode="rgb_array", height=512, width=512, camera_name="agentview")
            img_path = os.path.join(rollout_fig_path,
                                    "{:03}.png".format(t))
            imageio.imwrite(img_path, img)
        if env.is_success()["task"]:
            print('Rollout ' + str(i) + " is successful")

def rollout(
        policy,
        dataset_path,
        horizon,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        obs_keys=None,
        obs_len=1,
        num_rollouts=10):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        obs_keys (list): list of keys to use from the observation dictionary
        obs_len (int): number of observations to stack
        num_rollouts (int): number of rollouts to carry out
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """
    if video_writer is None:
        render_offscreen = False
    else:
        render_offscreen = True
    env = EnvUtils.create_env_from_metadata(
        env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path),
        render=False,            # no on-screen rendering
        # off-screen rendering to support rendering video frames
        render_offscreen=render_offscreen,
    )

    full_success = 0
    for i in tqdm(range(num_rollouts)):
        policy.set_eval()
        policy.reset()
        obs = env.reset()
        state_dict = env.get_state()

        # Required for deterministic action playback in robosuite tasks.
        obs = env.reset_to(state_dict)

        ob = TensorUtils.to_tensor(obs)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, device)
        ob = TensorUtils.to_float(ob)
        ob = torch.cat([value for key, value in ob.items()
                       if key in obs_keys], dim=-1)

        # initialize action and state deques
        state_deque = deque([ob] * obs_len, maxlen=obs_len)
        action_deque = deque()

        results = {}
        video_count = 0  # video frame counter
        total_reward = 0
        try:
            for step_i in range(horizon):

                policy_obs = torch.stack(list(state_deque), dim=0)
                if obs_len > 1:
                    policy_obs = torch.stack(list(state_deque), dim=1)

                if len(action_deque) == 0:
                    # get action from policy
                    act = policy.get_action(policy_obs)
                    if len(act.shape) == 3:
                        act = TensorUtils.to_numpy(act[0])
                    else:
                        act = TensorUtils.to_numpy(act)
                    for i in range(len(act)):
                        action_deque.append(act[i])

                act = action_deque.popleft()

                # play action
                next_obs, r, done, _ = env.step(act)

                # compute reward
                total_reward += r
                success = env.is_success()["task"]

                if video_writer is not None:
                    if video_count % video_skip == 0:
                        video_img = []
                        camera_names = ["agentview"]
                        for cam_name in camera_names:
                            video_img.append(env.render(
                                mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        # concatenate horizontally
                        video_img = np.concatenate(video_img, axis=1)
                        video_writer.append_data(video_img)
                    video_count += 1

                # break if done or if success
                if done or success:
                    break

                # update for next iter
                obs = deepcopy(next_obs)
                state_dict = env.get_state()

                ob = TensorUtils.to_tensor(obs)
                ob = TensorUtils.to_batch(ob)
                ob = TensorUtils.to_device(ob, device)
                ob = TensorUtils.to_float(ob)

                policy_obs = torch.cat(
                    [value for key, value in ob.items() if key in obs_keys], dim=-1)
                state_deque.append(policy_obs)

        except env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))

        stats = dict(Return=total_reward, Horizon=(
            step_i + 1), Success_Rate=float(success))
        full_success += float(success)

    return full_success/num_rollouts

def vlm_rollout(
    policy,
    dataset_path,
    horizon,
    vlm=None,
    instruction_path_list=[],
    subtask_ind=None,
    subtask_progress_key=None,
    generate_rollout_vid=None,
    generate_img=False,
    generate_vid=False,
    video_skip=5,
    img_skip=5,
    camera_names=None,
    obs_keys=None,
    obs_len=1,
    num_rollouts=1,
    figure_path="",
    video_path="",
    language_path="",
    verbose=False,
    ):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video,
    and returns the rollout trajectory.
    Args:
        policy (instance of RolloutPolicy): policy loaded from a checkpoint
        env (instance of EnvBase): env loaded from a checkpoint or demonstration metadata
        horizon (int): maximum horizon for the rollout
        render (bool): whether to render rollout on-screen
        video_writer (imageio writer): if provided, use to write rollout to video
        video_skip (int): how often to write video frames
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        obs_keys (list): list of keys to use from the observation dictionary
        obs_len (int): number of observations to stack
        num_rollouts (int): number of rollouts to carry out
    Returns:
        stats (dict): some statistics for the rollout - such as return, horizon, and task success
    """
    if generate_rollout_vid or generate_img or generate_vid:
        render_offscreen = True
    else:
        render_offscreen = False
    env = EnvUtils.create_env_from_metadata(
        env_meta=FileUtils.get_env_metadata_from_dataset(dataset_path),
        render=False,  # no on-screen rendering
        # off-screen rendering to support rendering video frames
        render_offscreen=render_offscreen,
    )
    if vlm is not None and len(instruction_path_list) > 1:
        vlm_instruction = read_text_file(instruction_path_list[0])
        if len(instruction_path_list) == 2:
            prompt1_instruction = read_text_file(instruction_path_list[1])
        elif len(instruction_path_list  ) == 3:            
            prompt1_instruction = read_text_file(instruction_path_list[1])
            prompt2_instruction = read_text_file(instruction_path_list[2])

    rollout_results = {}
    full_success = 0

    for rollout_i in tqdm(range(num_rollouts)):
        policy.set_eval()
        policy.reset()
        obs = env.reset()
        state_dict = env.get_state()
        rollout_fig_path = os.path.join(figure_path,
                                        'rollout_{:03}'.format(rollout_i))
        rollout_vid_path = os.path.join(video_path,
                                        'rollout_{:03}'.format(rollout_i))
        rollout_language_path= os.path.join(language_path,
                                        'rollout_{:03}'.format(rollout_i))
        os.makedirs(rollout_fig_path, exist_ok=True)
        os.makedirs(rollout_vid_path, exist_ok=True)
        os.makedirs(rollout_language_path, exist_ok=True)
        

        # Required for deterministic action playback in robosuite tasks.
        obs = env.reset_to(state_dict)

        ob = TensorUtils.to_tensor(obs)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, device)
        ob = TensorUtils.to_float(ob)
        ob = torch.cat([value for key, value in ob.items() if key in obs_keys],
                       dim=-1)

        # initialize action and state deques
        state_deque = deque([ob] * obs_len, maxlen=obs_len)
        action_deque = deque()

        # create a video writer
        video_writer = None
        if generate_rollout_vid:
            video_file = os.path.join(rollout_vid_path, "rollout.mp4")
            video_writer = imageio.get_writer(video_file, fps=20)
            if verbose:
                print(video_file)

        results = {'replan': [], 'subtask': [],
                   'reward': [], 'horizon': 0, 'success': []}
        total_reward = 0
        video_count = 0
        vlm_replan = False
        vlm_success = False
        current_subtask = 0
        img_path_list = []
        video_img = []
        try:
            for step_i in range(horizon):
                vlm_replan = False
                vlm_success = False

                # Save the image or video of the environment
                is_time_to_save_img = (step_i % img_skip == 0 or step_i == horizon - 1)
                is_time_to_save_vid =  (step_i % video_skip == 0 or step_i == horizon - 1)
                is_generate_img = generate_img and is_time_to_save_img
                is_generate_vid = generate_vid and is_time_to_save_vid
                if is_generate_img or is_generate_vid:
                    img = []
                    camera_names = ["agentview"]
                    for cam_name in camera_names:
                        img.append(
                            env.render(mode="rgb_array",
                                       height=512,
                                       width=512,
                                       camera_name=cam_name))
                    img = np.concatenate(img,
                                         axis=1)  # concatenate horizontally
                    video_img.append(img)
                    img_path = os.path.join(rollout_fig_path,
                                            "{:03}.png".format(step_i))
                    img_path_list.append(img_path)

                    # Generate video
                    if generate_vid:
                        video_file = os.path.join(
                            rollout_vid_path, "rollout_{:03d}.mp4".format(step_i))
                        if verbose:
                            print(video_file)
                        vlm_video_writer = imageio.get_writer(
                            video_file, fps=20)
                        camera_names = ["agentview"]

                        for img in video_img:
                            vlm_video_writer.append_data(img)
                        vlm_video_writer.close()

                    # Generate Image File
                    if generate_img:
                        if verbose:
                            print(img_path)
                        imageio.imwrite(img_path, img)  # Output the images

                policy_obs = torch.stack(list(state_deque), dim=0)
                if obs_len > 1:
                    policy_obs = torch.stack(list(state_deque), dim=1)

                env_success = policy_obs[..., -1,
                                         subtask_ind[current_subtask]].item() == 1
                policy_obs[:, :, subtask_ind[current_subtask]] = 0

                # Use VLM to check & verify the environment
                if vlm is not None and is_generate_img:
                    assert subtask_ind is not None
                    assert subtask_progress_key is not None
                    assert vlm_instruction is not None
                    
                    # Current time stamp: 125/496
                    vlm_instruction = vlm_instruction[:117] + str(
                        step_i) + '/' + str(horizon)
                    text_prompt = prompt1_instruction + vlm_instruction

                    if len(instruction_path_list) ==2 :
                        answer = vlm.get_response(
                        text_prompt=text_prompt, image_list=[img_path])
                    elif len(instruction_path_list) ==3:
                        prediction = vlm.get_response(text_prompt=text_prompt, image_list=[os.path.join(rollout_fig_path,
                                                                                                    "{:03}.png".format(0))])
                        text_prompt = prompt2_instruction + "\nPrediction:\n" + prediction
                        answer = vlm.get_response(
                        text_prompt=text_prompt, image_list=[img_path])

                    # Search in the VLM response for failure or success
                    if "failure" in answer.lower():
                        vlm_replan = True
                    elif "success" in answer.lower():
                        vlm_success = True
                    if current_subtask < len(subtask_ind) and  subtask_progress_key[current_subtask] in answer.lower():
                        vlm_success = True

                    
                    verbose1 = '----------Step {:03d}----------'.format(step_i)
                    verbose2 = 'Replanning: ' + str(vlm_replan) + "| Success: " + str(vlm_success) + '| Env Success: ' + str(env_success) + "| Current Subtask Index: " + str(current_subtask)
                    verbose3 = answer
                    lang_path = os.path.join(rollout_language_path,
                                            "{:03}.txt".format(step_i))
                    with open(lang_path, "w") as f:
                        f.write(verbose1 + '\n')
                        f.write(verbose2 + '\n')
                        f.write(verbose3)
                    # verbose_list.append([verbose1,verbose2,verbose3])
                    if verbose:
                        print(verbose1)
                        print(verbose2)
                        print(verbose3)

                    results['success'].append([vlm_success, env_success])
                    if vlm_success and env_success:
                        # Cap the number of subtasks
                        policy_obs[:, :, subtask_ind[current_subtask]] = 1
                        current_subtask = min(
                            current_subtask+1, len(subtask_ind))

                    if vlm_replan or (vlm_success and env_success):
                        policy.reset()
                        action_deque = deque()

                # Get action plan from policy
                if len(action_deque) == 0:
                    # get action from policy
                    act = policy.get_action(policy_obs)
                    if len(act.shape) == 3:
                        act = TensorUtils.to_numpy(act[0])
                    else:
                        act = TensorUtils.to_numpy(act)
                    for i in range(len(act)):
                        action_deque.append(act[i])

                    if (policy.is_plan_reset() or vlm_replan or step_i == 0) and step_i < horizon:
                        results['replan'].append(1)
                    else:
                        results['replan'].append(0)
                else:
                    results['replan'].append(0)

                act = action_deque.popleft()

                # play action
                next_obs, r, done, _ = env.step(act)

                # compute reward
                total_reward += r
                results['reward'].append(total_reward)
                success = env.is_success()["task"]

                # Recording Rollout Video
                if video_writer is not None:
                    if video_count % video_skip == 0:
                        rollout_video_img = []
                        camera_names = ["agentview"]
                        for cam_name in camera_names:
                            rollout_video_img.append(env.render(
                                mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        # concatenate horizontally
                        rollout_video_img = np.concatenate(
                            rollout_video_img, axis=1)
                        video_writer.append_data(rollout_video_img)
                    video_count += 1

                # break if done or if success
                if done or (success and vlm_success):
                    if subtask_ind is not None:
                        results['subtask'].append(len(subtask_ind))
                    break

                # update for next iter
                obs = deepcopy(next_obs)
                state_dict = env.get_state()

                ob = TensorUtils.to_tensor(obs)
                ob = TensorUtils.to_batch(ob)
                ob = TensorUtils.to_device(ob, device)
                ob = TensorUtils.to_float(ob)

                policy_obs = torch.cat(
                    [value for key, value in ob.items() if key in obs_keys],
                    dim=-1)

                if subtask_ind is not None:
                    start_ind = subtask_ind[0]
                    end_ind = subtask_ind[-1]+1
                    results['subtask'].append(
                        float(policy_obs[..., start_ind:end_ind].sum().cpu()))
                    if vlm is not None:
                        start_ind = subtask_ind[0]
                        curr_ind = subtask_ind[current_subtask]
                        end_ind = subtask_ind[-1]+1
                        policy_obs[..., start_ind:curr_ind] = 1
                        policy_obs[..., curr_ind:end_ind] = 0
                state_deque.append(policy_obs)

        except env.rollout_exceptions as e:
            print("WARNING: got rollout exception {}".format(e))

        stats = dict(Return=total_reward,
                     Horizon=(step_i + 1),
                     Success_Rate=float(success))
        full_success += float(success)
        results['horizon'] = step_i + 1
        rollout_results.update({rollout_i: results})

    return full_success / num_rollouts, stats, rollout_results

def process_batch(batch):
    """
    Process a batch of data.
    """
    input_batch = dict()
    # concat all the items in batch["obs"]
    input_batch["obs"] = torch.cat(
        [value for value in batch['obs'].values()], dim=-1)

    input_batch["actions"] = batch["actions"]
    return TensorUtils.to_float(TensorUtils.to_device(input_batch, device))

def run_epoch(model, data_loader, validate=False):
    """
    Run a single epoch of training.
    """
    data_loader_iter = iter(data_loader)
    if validate:
        model.eval()
    else:
        model.train()
    total_loss = 0
    for batch_i, batch in enumerate(data_loader_iter):
        inputs = process_batch(batch)
        loss = model.train_on_batch(inputs, validate)
        total_loss += loss
    return total_loss / len(data_loader)

def train(model, train_loader, valid_loader=None, start_epoch=0, num_epochs=100, save_path=None):
    """
    Train a model using the algorithm.
    """
    train_losses = []
    valid_losses = []
    for epoch_i in tqdm(range(start_epoch, start_epoch+num_epochs)):

        train_loss = run_epoch(model, train_loader)
        train_losses.append((train_loss, model.epoch))

        if valid_loader is not None:
            valid_loss = run_epoch(model, valid_loader, validate=True)
            valid_losses.append((valid_loss, model.epoch))
        if epoch_i % 10 == 0:
            print("Epoch: {} Train Loss: {} Valid Loss: {}".format(
                epoch_i, train_loss, valid_loss))
        if epoch_i % 10 == 0:
            model.save(os.path.join(
                save_path, "epoch_{}.pth".format(model.epoch)))
        model.epoch += 1
    return train_losses, valid_losses

def load_data_for_training(dataset_path, obs_keys, seq_len=1, batch_size=100, normalize=False, frame_stack=1):
    """
    Load data for training.
    """
    action_keys = ("actions",)
    action_config = {"actions": {}}

    train_set = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=action_keys,
        dataset_keys=["actions"],
        action_config=action_config,
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",
        hdf5_use_swmr=False,
        hdf5_normalize_obs=normalize,
        filter_by_attribute='train',
    )
    valid_set = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=action_keys,
        dataset_keys=["actions"],
        action_config=action_config,
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",
        hdf5_use_swmr=False,
        hdf5_normalize_obs=normalize,
        filter_by_attribute='valid',
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

    train_batch = next(iter(train_loader))
    print("batch keys: {}".format(train_batch.keys()))
    print("observation shapes: ")
    for obs, obs_key in train_batch["obs"].items():
        print("{} shape: {}".format(obs, obs_key.shape))
    print("action shape: {}".format(train_batch['actions'].shape))

    return train_loader, valid_loader
