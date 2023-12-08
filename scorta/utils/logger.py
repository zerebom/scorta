import inspect
import os
import sys
import time
from functools import wraps, partial
from typing import Any, Callable, TypeVar

from loguru import logger

T = TypeVar("T", bound=Callable[..., Any])


def relative_path(full_path: str) -> str:
    try:
        return os.path.relpath(full_path)
    except ValueError:
        return full_path


# loguruのフォーマットを変えてる
logger.remove(0)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level:5}| <blue>{file}:{function}:{line}</blue> | <level>{message}</level>",
    level="INFO",
)


def timing_logger(func=None, *, log_arg: str | None = None):
    """
    各メソッドにどれくらい時間がかかったわかるようにするデコレーター
    log_arg を指定すると、引数の値もログに出力する
    """
    if func is None:
        return partial(timing_logger, log_arg=log_arg)  # type: ignore

    @wraps(func)
    def wrapper(*args, **kwargs):
        pid = str(os.getpid())[-5:]
        start_time = time.perf_counter()
        frame = inspect.stack()[1]
        filename = relative_path(frame.filename)
        lineno = frame.lineno

        log_message = f"[PID:{pid}] Starting {func.__name__}() at {filename}:{lineno}"
        if log_arg is not None:
            log_message += f" with arg: {log_arg}"
        logger.info(log_message)

        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        logger.info(f"[PID:{pid}] Finished {func.__name__}() in {elapsed_time:.4f} seconds.\n")

        return result

    return wrapper  # type: ignore
