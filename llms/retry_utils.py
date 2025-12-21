import concurrent.futures
import functools
import random
import time
from concurrent.futures import TimeoutError
from typing import Any, Callable

from core_utils.signal_utils import signal_manager

_EXECUTOR_REGISTRY: dict[str, concurrent.futures.ThreadPoolExecutor] = {}


def cleanup_executors() -> None:
    print(f"{__name__}: Cleaning up threadpool executors")
    for key in _EXECUTOR_REGISTRY:
        try:
            _EXECUTOR_REGISTRY[key].shutdown(wait=False)
            print(f"{__name__}: Threadpool executors cleaned up")
        except Exception as e:
            print(f"{__name__}: Error shutting down executor for {key}: {e}")


signal_manager.add_cleanup_function(cleanup_executors)


def get_executor(key: str, max_workers: int = 1) -> concurrent.futures.ThreadPoolExecutor:
    if key not in _EXECUTOR_REGISTRY:
        _EXECUTOR_REGISTRY[key] = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        print(f"Created new executor for {key}")
    return _EXECUTOR_REGISTRY[key]


def retry_with_exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60,
    exp_base: float = 2,
    jitter: bool = True,
    max_retries: int = 2,
    api_errors: tuple[type[Exception], ...] = (),
    custom_errors: tuple[type[Exception], ...] = (),
    timeout_getter: Callable | None = None,  # type: ignore
    handle_custom_errors: Callable | None = None,  # type: ignore
    handle_api_errors: Callable | None = None,  # type: ignore
    handle_max_retries: Callable | None = None,  # type: ignore
    max_workers: int = 1,
    id_getter: Callable | None = None,  # type: ignore
    logger: Any | None = None,
) -> Callable[..., Any]:
    """Retry a function with exponential backoff.

    Args:
        base_delay (float): Initial delay before the first retry.
        max_delay (float): Maximum delay between retries.
        exp_base (float): Exponential factor to increase delay.
        jitter (bool): If True, adds randomness to the delay.
        max_retries (int): Maximum number of retries before handling max retries.
        api_errors (tuple): Tuple of API error classes to catch.
        custom_errors (tuple): Tuple of custom error classes to catch.
        timeout_getter (Callable | None): Callable that returns how long to wait func to return before retry. Default is no timeout.
        handle_custom_errors (Callable | None): Function to handle custom errors.
        handle_api_errors (Callable | None): Function to handle API errors.
        handle_max_retries (Callable | None): Function to handle the case when max_retries is exceeded.
        max_workers (int): Number of workers for the executor.

    Raises:
        e: Exception raised by the function.
        err: Error raised by the function.
        err: Error raised by the function.

    Returns:
        Any: Result of the function.
    """
    if timeout_getter is None:
        timeout_getter = lambda args, kwargs: None
    if id_getter is None:
        id_getter = lambda args, kwargs: ""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create a unique key from module and function name.
        func_name = func.__name__
        module_name = func.__module__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            num_retries = 0
            delay = base_delay
            timeout = timeout_getter(args, kwargs)

            unique_key = f"{module_name}.{func_name}{id_getter(args, kwargs)}"

            executor = get_executor(unique_key, max_workers=max_workers)

            while True:
                should_retry, apply_delay, increment_retries = True, True, True
                try:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=timeout)

                except Exception as e:
                    if isinstance(e, TimeoutError):
                        future.cancel()

                    # Use provider-specific error handling if provided.
                    if api_errors and isinstance(e, api_errors) and handle_api_errors:
                        err, should_retry, apply_delay, increment_retries = handle_api_errors(e, *args, **kwargs)

                    elif custom_errors and isinstance(e, custom_errors) and handle_custom_errors:
                        err, should_retry, apply_delay, increment_retries = handle_custom_errors(e, *args, **kwargs)

                    else:
                        raise e

                    if not should_retry:
                        raise err

                    num_retries += increment_retries
                    if num_retries <= max_retries:
                        if apply_delay:
                            delay = (delay + (jitter * random.random())) * exp_base
                            delay = min(delay, max_delay)
                            if logger:
                                logger.info(f"Retrying in {delay} seconds")
                            else:
                                print(f"Retrying in {delay} seconds", flush=True)
                            time.sleep(delay)
                        continue
                    else:
                        if handle_max_retries:
                            err, should_retry, _, _ = handle_max_retries(e, *args, **kwargs)
                            if not should_retry:
                                raise err
                        num_retries = 0
                        delay = base_delay

        return wrapper

    return decorator
