import time
import sys
import json


def warn(s):
    print(s, file=sys.stderr)


def split_list(l, n):
    return [l[i : i + n] for i in range(0, len(l), n)]


class Timer:
    def __enter__(self, name=""):
        self.start = time.perf_counter()
        self.name = name
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        warn(f"{self.name} timer (sec): {str(self.interval)}")

