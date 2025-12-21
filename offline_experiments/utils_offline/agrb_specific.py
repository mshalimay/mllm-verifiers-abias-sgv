from agrb.utils_agrb.vwa_trajectory import VWATrajectoryView


def get_trace_data_agrb(trajectory_path: str, env: str, img_ann_types: list[str] = []):
    trajectories_dir = trajectory_path.split("trajectories/")[0] + "trajectories"

    if "vwa" in env:
        traj = VWATrajectoryView(
            trajectory_path,
            trajectories_dir=trajectories_dir,
            ann_types=img_ann_types,
            update_intent=True,
        )
        traj.add_data_vwa_format()
        return {
            "objective": traj.objective,
            "trajectory": traj,
            "meta_data": traj.meta_data,
            "task_id": str(traj.task_id),
            "domain": traj.vwa_domain,
            "vwa_task_id": traj.vwa_task_id,
            "trajectory_path": trajectory_path,
        }

    else:
        raise NotImplementedError(f"Environment {env} not supported")
