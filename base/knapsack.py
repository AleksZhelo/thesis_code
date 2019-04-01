import collections
import functools


# noinspection PyPep8Naming
class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def knapsack(items, maxweight):
    """
    Modified from https://codereview.stackexchange.com/a/20581

    Solve the knapsack problem by finding the most valuable
    subsequence of `items` subject that weighs no more than
    `maxweight`, where the value is equal to the weight.

    `items` is a sequence of pairs `(any, weight)`, where `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of weights in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [('a', 12), ('b', 1), ('c', 4), ('d', 1), ('e', 2)]
    >>> knapsack(items, 15)
    (15, [('a', 12), ('b', 1), ('e', 2)])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    # noinspection SpellCheckingInspection
    @memoized
    def bestvalue(i, j):
        if i == 0:
            return 0
        _, weight = items[i - 1]
        value = weight
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    for i in range(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result.append(items[i - 1])
            j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result
