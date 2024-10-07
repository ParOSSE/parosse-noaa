from parmap.parmap import Parmap
from math import pow


def cubed(i):
    return pow(i, 3)


def test():
    items = list(range(1, 4))
    for mode in ["seq", "par"]:
        parmap = Parmap(mode, numWorkers=4)
        res = parmap(fn=cubed, items=items)
        print(res)


if __name__ == '__main__':
    test()
