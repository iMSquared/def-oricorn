import time
from contextlib import contextmanager


def time_stamp() -> str:
    """Get the unqiue time stamp for a process or file"""
    from datetime import datetime
    now = datetime.now()
    current_date = now.date()
    month = current_date.month
    day = current_date.day
    current_time = now.time()
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    millis = current_time.microsecond/1000
    TIME_STAMP = f"data_{month}m{day}d{hour}h{minute}m{second}s{f'{millis}'.zfill(3)}ms"

    return TIME_STAMP



def profile_time(function):
    def wrapper(*args, **kwargs):
        # Profiler start
        # print(f"{function.__name__} start")
        check_time = time.time()
        # Some function...
        result = function(*args, **kwargs)
        # Profiler ends
        check_time = time.time() - check_time
        print(f"{function.__name__} end: {check_time:.4f}(s)")
        

        return result
    return wrapper


@contextmanager
def profile_context(context: str):
    check_time = time.time()
    yield
    check_time = time.time()-check_time
    print(f"{context} end: {check_time:.4f}(s)")