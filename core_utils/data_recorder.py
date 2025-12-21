import json
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd


class DataRecorder:
    def __init__(
        self,
        result_dir: str,
        config_file_list: list[dict[str, Any]],
        test_config_json_file: str,
    ):
        self.result_dir = Path(result_dir)
        self.total_num_tasks = len(config_file_list)
        self.num_failed_executions = 0
        self.test_config_json_file = test_config_json_file

        # Initialize data structures
        self.summary_data = {}
        self.data = {}
        self.failed_task_ids = set()
        self.unfinished_task_ids = {f"{task['domain_task_id']}_{task['domain']}_{task['env']}" for task in config_file_list}

        # Load existing data if resuming experiment
        self._load_existing_data()
        self._load_failed_tasks()

    def _load_failed_tasks(self):
        if (self.result_dir / "failed_tasks.txt").exists():
            with open(self.result_dir / "failed_tasks.txt", "r") as f:
                for line in f.readlines():
                    try:
                        self.failed_task_ids.add(line.strip())
                    except ValueError:
                        continue

    def _load_existing_data(self):
        if (self.result_dir / "summary_data.csv").exists():
            df = pd.read_csv(self.result_dir / "summary_data.csv")
            if "attempt_id" not in df.columns:
                df["attempt_id"] = 0
            # Ensure correct dtypes
            if "task_id" in df.columns:
                try:
                    df["task_id"] = df["task_id"].astype(str)
                except Exception:
                    pass
            try:
                df["attempt_id"] = df["attempt_id"].astype(int)
            except Exception:
                pass

            # Build self.data from CSV rows
            for _, row in df.iterrows():
                task_id_val = str(row["task_id"]) if "task_id" in row else str(_)
                domain_val = str(row["domain"]) if "domain" in row else ""
                env_val = str(row["env"]) if "env" in row else ""
                attempt_id_val = int(row["attempt_id"]) if "attempt_id" in row else 0
                unique_id = f"{task_id_val}_{domain_val}_{env_val}"

                if unique_id not in self.data:
                    sites_val: list[str] = []
                    if "sites" in df.columns and pd.notna(row.get("sites")):
                        try:
                            sites_val = json.loads(row["sites"]) if isinstance(row["sites"], str) else []
                        except Exception:
                            sites_val = []
                    self.data[unique_id] = {
                        "attempts": {},
                        "env": env_val,
                        "domain": domain_val,
                        "sites": sites_val,
                    }

                attempt_entry: dict[str, Any] = {}
                if "traj_json_file" in df.columns and pd.notna(row.get("traj_json_file")):
                    attempt_entry["traj_json_file"] = str(row["traj_json_file"])  # type: ignore
                if "score" in df.columns and pd.notna(row.get("score")):
                    attempt_entry["score"] = float(row["score"])  # type: ignore
                if "num_actions" in df.columns and pd.notna(row.get("num_actions")):
                    attempt_entry["num_actions"] = int(row["num_actions"])  # type: ignore
                if "elapsed_time" in df.columns and pd.notna(row.get("elapsed_time")):
                    attempt_entry["elapsed_time"] = float(row["elapsed_time"])  # type: ignore

                # Defaults
                if "elapsed_time" not in attempt_entry:
                    attempt_entry["elapsed_time"] = np.nan
                if "score" not in attempt_entry:
                    attempt_entry["score"] = np.nan
                if "traj_json_file" not in attempt_entry:
                    attempt_entry["traj_json_file"] = ""

                self.data[unique_id]["attempts"][attempt_id_val] = attempt_entry

    def initialize_task(
        self,
        task_id: int | str,
        domain: str,
        env: str,
        sites: List[str],
        attempt_id: int = 0,
        traj_json_file: str = "",
    ):
        """Initialize data recording structures for a new task attempt"""
        unique_id = f"{task_id}_{domain}_{env}"
        if unique_id not in self.data:
            self.data[unique_id] = {
                "attempts": {},
                "env": env,
                "domain": domain,
                "sites": sites,
            }
        self.data[unique_id]["attempts"][attempt_id] = {
            "elapsed_time": np.nan,
            "score": np.nan,
            "traj_json_file": traj_json_file,
        }

    def update_save_data(
        self,
        task_id: int | str,
        domain: str,
        env: str,
        score: float,
        elapsed_time: float,
        num_actions: int,
        attempt_id: int = 0,
    ):
        """Update all data and summary statistics for a task attempt"""
        unique_id = f"{task_id}_{domain}_{env}"

        # Record score, number of actions, elapsed time
        if unique_id not in self.data:
            self.data[unique_id] = {
                "attempts": {},
                "env": env,
                "domain": domain,
                "sites": [],
            }
        if attempt_id not in self.data[unique_id]["attempts"]:
            self.data[unique_id]["attempts"][attempt_id] = {
                "elapsed_time": np.nan,
                "score": np.nan,
                "traj_json_file": "",
            }

        self.data[unique_id]["attempts"][attempt_id]["score"] = score
        self.data[unique_id]["attempts"][attempt_id]["num_actions"] = num_actions
        self.data[unique_id]["attempts"][attempt_id]["elapsed_time"] = elapsed_time

        self.save_to_disk()

    def save_execution_summary(
        self,
        total_time: float,
        provider: str = "",
    ):
        """Save summary of the entire execution"""
        execution_data = [
            {
                "n_tasks": self.total_num_tasks,
                "n_failed_executions": self.num_failed_executions,
                "total_time": total_time,
                "provider": provider,
            }
        ]

        # Save to temp file + rename to avoid race conditions
        dest_file_csv = self.result_dir / "execution_data.csv"
        tmp_csv_file = dest_file_csv.with_suffix(".tmp")
        pd.DataFrame(execution_data).to_csv(tmp_csv_file, index=False)
        tmp_csv_file.rename(dest_file_csv)

    def save_to_disk(self):
        """Save current data to disk"""

        dest_file_csv = self.result_dir / "summary_data.csv"
        tmp_csv_file = dest_file_csv.with_suffix(".tmp")
        rows: list[dict[str, Any]] = []

        for unique_id, group in self.data.items():
            # Safer split in case task_id contains underscores
            try:
                task_id, domain_from_id, env_from_id = unique_id.rsplit("_", 2)
            except ValueError:
                # Fallback to stored values
                task_id = unique_id
                domain_from_id = group.get("domain", "")
                env_from_id = group.get("env", "")

            for attempt_id, attempt_data in group.get("attempts", {}).items():
                if "score" not in attempt_data:
                    continue
                row = {
                    "task_id": task_id,
                    "domain": domain_from_id,
                    "env": env_from_id,
                    "attempt_id": attempt_id,
                    "traj_json_file": attempt_data.get("traj_json_file", ""),
                    "score": attempt_data.get("score", np.nan),
                    "num_actions": attempt_data.get("num_actions", np.nan),
                    "elapsed_time": attempt_data.get("elapsed_time", np.nan),
                    "sites": json.dumps(group.get("sites", [])),
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(tmp_csv_file, index=False)
        tmp_csv_file.rename(dest_file_csv)

    @staticmethod
    def _calculate_stats(data: list) -> dict:
        """Calculate basic statistics for a list of numbers"""
        try:
            stats = {"sum": sum(data), "avg": sum(data) / len(data), "max": max(data), "min": min(data)}
        except ZeroDivisionError:
            stats = {"sum": None, "avg": None, "max": None, "min": None}
        return stats

    def _update_unfinished_failed_tasks(self, unique_id: str, task_success: bool, task_set: set, file_path: Path):
        initial_len = len(task_set)
        if task_success:
            task_set.discard(unique_id)
        else:
            task_set.add(unique_id)

        if initial_len != len(task_set):
            with open(file_path, "w") as f:
                f.write(self.test_config_json_file + "\n")
                for value in sorted(task_set):
                    f.write(f"{value}\n")

    def update_unfinished_failed_tasks(self, task_id: int | str, task_success: bool, domain: str, env: str):
        # unique_id = f"{task_id}_{domain}_{env}"
        unique_id = f"{task_id}_{domain}"
        self._update_unfinished_failed_tasks(unique_id, task_success, self.failed_task_ids, self.result_dir / "failed_tasks.txt")
        self._update_unfinished_failed_tasks(unique_id, task_success, self.unfinished_task_ids, self.result_dir / "unfinished_tasks.txt")

    def get_scores(self) -> list[float]:
        return [attempt_data["score"] for unique_id in self.data for attempt_id, attempt_data in self.data[unique_id]["attempts"].items() if "score" in attempt_data]
