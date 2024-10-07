"""
parmap.py -- Utility function that does a parallel Map (map/collect) of a Function onto a list of Items
Degree of parallelism is controlled by numPartitions parameter.
Parallel framework is chosen by mode parameter.
"""
import sys
import multiprocessing
from parmap.utils import Timer, warn


# Legal parallel execution modes
MODES = [
    "seq",  # sequential
    "par",  # pool.map using multiprocessing library
]


def warn(s):
    print(s, file=sys.stderr)


# def warn(s): print >>sys.stderr, s


class Parmap:
    """Do a parallel Map of a function onto a list of items and then collect and return the list of results.
    The function receives a list of values or URL's and can perform I/O as a side effect (e.g. write files/objects and return their URLs).
    """

    def __init__(
        self,
        mode="par",  # one of the six modes above
        # number of workers, degree of parallelism (None means auto)
        numWorkers=None,
        master=None,  # URL for master scheduler
        # configuration dict for additional info:  credentials, etc.
        config={},
        context=None,  # reuse existing scheduler context
    ):
        if mode not in MODES:
            warn(
                'parmap: Bad mode arg, using "par" (local multicore) instead: %s' % mode
            )
            mode = "par"
        self.mode = mode
        self.numWorkers = numWorkers
        self.master = master
        self.config = config
        self.context = context

        if mode == "seq":
            pass

        elif mode == "par":
            if self.context is None:
                worker = config.get("worker", None)
                if worker is not None and worker == "thread":
                    self.context = multiprocessing.pool.ThreadPool(numWorkers)
                else:
                    self.context = multiprocessing.Pool(numWorkers)

    def __call__(
        self,
        fn,
        items,
        numPartitions=None,
    ):
        if numPartitions is None:
            numPartitions = self.numWorkers
        mode = self.mode
        ctx = self.context
        warn(
            "\nparmap %s: Running in mode %s with numPartitions %s"
            % (str(fn), mode, str(numPartitions))
        )

        if mode == "seq":
            return list(map(fn, items))

        elif mode == "par":
            return ctx.map(fn, items)  # here a Pool of processes


def test(fn, n):
    from math import pow

    items = list(range(1, n))
    for mode in ["seq", "par"]:
        parmap = Parmap(mode, numWorkers=4, config={"appName": "parmap_factorial_test"})
        with Timer():
            results = parmap(fn, items, numPartitions=4)
        print(results[-1])


def cubed(i):
    from math import pow

    return pow(i, 3)


# if __name__ == "__main__":
#     from math import factorial

#     fn = sys.argv[1]
#     n = int(sys.argv[2])
#     if fn == "cubed":
#         fn = cubed
#     else:
#         fn = factorial
#     test(fn, n)


# python parmap.py cubed 100000
# python parmap.py factorial 10000
