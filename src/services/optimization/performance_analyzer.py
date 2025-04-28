import time

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Function {func.__name__} took {elapsed:.3f} sec")
        return out
    return wrapper