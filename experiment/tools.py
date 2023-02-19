import time
from functools import wraps
import numpy as np


class FpsCalc:
    time_results = {}

    @classmethod
    def calc_time(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed_time = end - start
            if not func.__name__ in self.time_results:
                self.time_results[func.__name__] = []
            self.time_results[func.__name__].append(elapsed_time)
            return result
        return wrapper

    @classmethod
    def get_results(self):
        results = {}
        for key in self.time_results:
            time_list = np.array(self.time_results[key])
            mean_time = time_list.mean()
            results[key] = mean_time
        return results


@FpsCalc.calc_time
def f(x):
    y = 0
    for i in range(10000):
        y += x
    return y

if __name__ == '__main__':
    for x in range(1000):
        y = f(x)
    results = FpsCalc.get_results()
    print(results)