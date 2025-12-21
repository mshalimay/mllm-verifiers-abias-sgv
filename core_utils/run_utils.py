import os
import signal
import subprocess

from core_utils.signal_utils import signal_manager


def _terminate_process_group(process: subprocess.Popen, timeout: float = 5.0):
    """Terminate the entire process group started for `process`.

    Sends SIGTERM to the group, waits up to `timeout` seconds, and escalates to SIGKILL if needed.
    Safe to call multiple times.
    """
    try:
        # When start_new_session=True, the child becomes a new session and process group leader (PGID=PID)
        pgid = os.getpgid(process.pid)
    except Exception:
        # If we cannot get a pgid, fall back to terminating just the child
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except Exception:
        # Best-effort; don't raise
        pass

    try:
        process.wait(timeout=timeout)
        return
    except Exception:
        # Still alive; escalate
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=timeout)
        except Exception:
            pass


def run_and_wait(command, logfile_path):
    if logfile_path:
        with open(logfile_path, "w") as logfile:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,  # Make child the leader of a new process group for clean teardown
            )
            # Ensure termination tears down the whole process group
            signal_manager.register_termination_signals(lambda: _terminate_process_group(process))  # type:ignore
            while True:
                output_line = process.stdout.readline()  # type:ignore
                if output_line == "" and process.poll() is not None:
                    break
                if output_line:
                    print(output_line, end="")  # Print to command line
                    logfile.write(output_line)
                    logfile.flush()
            process.wait()
    else:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # Make child the leader of a new process group for clean teardown
        )
        # Ensure termination tears down the whole process group
        signal_manager.register_termination_signals(lambda: _terminate_process_group(process))  # type:ignore
        while True:
            output_line = process.stdout.readline()  # type:ignore
            if output_line == "" and process.poll() is not None:
                break
            if output_line:
                print(output_line, end="")  # Print to command line
        process.wait()
