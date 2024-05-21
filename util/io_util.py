'''
For low level IO
'''

from dataclasses import dataclass
import datetime
from pathlib import Path
import pandas as pd
import numpy.typing as npt
from typing import List, Tuple


import time
from contextlib import contextmanager



@dataclass
class ProfilingNamespace:
    start_time: float
    end_time: float = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def profile_time(function, print_log=False):
    """Some custom decorator for profiling."""
    def wrapper(*args, **kwargs):
        # Profiler start
        # print(f"{function.__name__} start")
        check_time = time.time()
        # Some function...
        result = function(*args, **kwargs)
        # Profiler ends
        duration = time.time() - check_time
        if print_log:
            print(f"{function.__name__} end: {duration:.3f}(s)")
        
        return result, duration
    return wrapper


@contextmanager
def context_profiler(context: str, print_log=False):
    data = ProfilingNamespace(start_time = time.time())
    try:
        yield data
    finally:
        data.end_time = time.time()
        if print_log:
            print(f"[Profiler] \"{context}\" elapsed time: {data.duration:.3f} seconds")



@dataclass(frozen=True)
class BuildMetadata:
    max_dec_num: int
    max_vertices: int
    vol_error_tol: int
    vhacd_ver: str
    build: str

    @classmethod
    def from_str(cls, build: str) -> "BuildMetadata": 
        """Parse build"""
        splited = build.split("_")
        build_metadata = cls(
            max_dec_num = int(splited[0]),
            max_vertices = int(splited[1]),
            vol_error_tol = int(splited[2]),
            vhacd_ver = splited[3],
            build = build)
        return build_metadata
    
    def __str__(self):
        return self.build
    

def get_filepath_from_annotation(
        annotation_path: str,
        obj_base_dir: str, 
        prefixes: List[str] = ["32_64_1_v4", "32_64_5_v4"]
) -> List[str]:
    """Get full path to the file to use"""
    with Path(annotation_path).open("rb") as f:
        df = pd.read_csv(f)
    filename_list = df["name"].values.tolist()

    prefixed = []
    for prefix in prefixes:
        for fn in filename_list:
            new_name = prefix+fn
            prefixed.append(new_name)

    full_paths = []
    for fn in prefixed:
        path = str(Path(obj_base_dir)/fn)
        full_paths.append(path)

    return full_paths