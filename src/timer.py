import signal
from contextlib import contextmanager
import time

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds=60):
    """
    Context manager to limit the time of a function call. default is 60s
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def long_function_call():
    # time.sleep(15)
    while True:
        pass

def main():
    try:
        begin = time.time()
        with time_limit(60):
            long_function_call()
    except TimeoutException as e:
        elapsed = time.time() - begin
        print(f"After {elapsed}s, timed out!")

if __name__ == "__main__":
    main()