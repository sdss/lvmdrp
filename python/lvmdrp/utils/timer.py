import time
import tracemalloc
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator
    Use as
        Timer t
        t.start()
        code() 
        t.stop()

        or
        @Timer()
        def func(...)

        or
        with Timer(name):
            code()
        
        see https://realpython.com/python-timer/
    """

    timers: ClassVar[Dict[str, float]] = {}
    name: Optional[str] = None
    text: str = "elapsed time: {:0.4f} seconds using {:.2f} MB of memory (peak)"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        tracemalloc.start()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        _, peak = tracemalloc.get_traced_memory()
        peak = peak / (1024**2) # in MB
        tracemalloc.stop()  # Stop memory tracking

        # Report elapsed time
        if self.logger:
            if self.name is not None:
                self.logger(self.name + ': ' + self.text.format(elapsed_time, peak))
            else:
                self.text.format(elapsed_time, peak)
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()

