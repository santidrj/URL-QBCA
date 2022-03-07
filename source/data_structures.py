import heapq


class MaxHeapObj:
    """
    Max-Heap object.

    This class redefines the lower than operator in order to emulate the behavior of a Max-Heap using the heapq library.
    """

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val > other.val

    def __eq__(self, other):
        return self.val == other.val

    def __str__(self):
        return str(self.val)


class MaxHeap(object):
    """
    Max-Heap data structure.
    """

    def __init__(self):
        self.h = []

    def heappush(self, x):
        heapq.heappush(self.h, MaxHeapObj(x))

    def heappop(self):
        return heapq.heappop(self.h).val

    def __getitem__(self, i):
        return self.h[i].val

    def __len__(self):
        return len(self.h)
