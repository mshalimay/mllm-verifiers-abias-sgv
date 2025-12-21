import atexit
import os
import signal
from typing import Any, Callable


class SignalManager:
    def __init__(self) -> None:
        self._cleanup_functions: list[Callable[[], None]] = []
        self._signals_registered: bool = False
        self.logger: Any | None = None

    def add_cleanup_function(self, cleanup_func: Callable[[], None]) -> None:
        """
        Adds a cleanup function and automatically registers the termination signals
        if they haven't been registered already.
        """
        self._cleanup_functions.append(cleanup_func)
        if not self._signals_registered:
            self.register_termination_signals(register_atexit=True)
            self._signals_registered = True

    def print_or_log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        else:
            try:
                print(message, flush=True)
            except OSError:
                # Silently handle I/O errors during termination
                pass

    def composite_handler(self, signum: int, frame: Any) -> None:
        try:
            self.print_or_log(f"Received termination signal {signum}. Cleaning up...")
            for func in self._cleanup_functions:
                try:
                    func()
                except Exception as e:
                    self.print_or_log(f"Error during cleanup in {func.__name__}: {e}")
            self.print_or_log("Exiting...")
        except Exception:
            # Ensure we exit even if logging fails
            pass
        finally:
            os._exit(1)

    def register_termination_signals(self, register_atexit: bool = False) -> None:
        termination_signals: list[int] = []
        if hasattr(signal, "SIGINT"):
            termination_signals.append(signal.SIGINT)
        if hasattr(signal, "SIGTERM"):
            termination_signals.append(signal.SIGTERM)
        if hasattr(signal, "SIGHUP"):
            termination_signals.append(signal.SIGHUP)
        if hasattr(signal, "SIGQUIT"):
            termination_signals.append(signal.SIGQUIT)
        if hasattr(signal, "SIGUSR1"):
            termination_signals.append(signal.SIGUSR1)
        if hasattr(signal, "SIGUSR2"):
            termination_signals.append(signal.SIGUSR2)

        if register_atexit:
            # Register with atexit so cleanup is attempted on normal termination.
            atexit.register(lambda: [func() for func in self._cleanup_functions])

        # Register the composite handler for each termination signal.
        for sig in termination_signals:
            signal.signal(sig, self.composite_handler)

    def set_logger(self, logger: Any) -> None:
        self.logger = logger


# Create a singleton instance to be used throughout your application.
signal_manager = SignalManager()


# Deprecated
# def handle_termination_signal(fun_to_register, signum, frame, logger=None):
#     """
#     Common signal handler that calls the provided cleanup function and then exits.
#     """
#     if logger:
#         logger.info(f"Received termination signal {signum}. Cleaning up...")
#     else:
#         print(f"Received termination signal {signum}. Cleaning up...", flush=True)

#     try:
#         fun_to_register()
#     except Exception as e:
#         if logger:
#             logger.error(f"Error during cleanup: {e}")
#         else:
#             print(f"Error during cleanup: {e}", flush=True)
#     os._exit(1)
