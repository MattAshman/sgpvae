__all__ = ['step_function']


def step_function(x):
    x[x <= 0] = 0
    x[x > 0] = 1

    return x
