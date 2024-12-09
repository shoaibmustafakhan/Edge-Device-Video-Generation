# memory_utils.py
import psutil
import os
import logging
from functools import wraps
import time

logger = logging.getLogger(__name__)

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def memory_tracker(func):
    """Decorator to track memory usage before and after function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Memory before {func.__name__}:")
        print_memory_usage()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Memory after {func.__name__}:")
        print_memory_usage()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        
        return result
    return wrapper