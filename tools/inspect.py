import time
import random

class FPSCalculator:
    DICT = {}

    @staticmethod
    def start(name):
        FPSCalculator.DICT.setdefault(name, {'start':[], 'end':[], 'time':[]})
        if len(FPSCalculator.DICT[name]['start']) == len(FPSCalculator.DICT[name]['end']):
            t = time.time()
            FPSCalculator.DICT[name]['start'].append(t)
        else:
            print(f"{name} has already started")

    @staticmethod
    def end(name):        
        if name not in FPSCalculator.DICT.keys():
            return 
        if len(FPSCalculator.DICT[name]['end']) == len(FPSCalculator.DICT[name]['start']) - 1:
            t = time.time()
            FPSCalculator.DICT[name]['end'].append(t)
            elapsed_time = t - FPSCalculator.DICT[name]['start'][-1]
            FPSCalculator.DICT[name]['time'].append(elapsed_time)
        else:
            print(f"{name} has not started")

    @staticmethod
    def get_execution_time(name, duration=1):
        if name not in FPSCalculator.DICT.keys():
            return 0
        count = len(FPSCalculator.DICT[name]['time'])
        if count == 0:
            return 0
        elif count >= duration:
            return sum(FPSCalculator.DICT[name]['time'][-duration:]) / duration
        else:
            return sum(FPSCalculator.DICT[name]['time']) / count

    @ staticmethod
    def calc_fps(func):
        def wrapper(*args, **kwargs):
            FPSCalculator.start(func.__name__)
            res = func(*args, **kwargs)
            FPSCalculator.end(func.__name__)
            return res
        return wrapper


@FPSCalculator.calc_fps
def func():
    t = random.random()
    time.sleep(t)
    return

if __name__ == '__main__':

    for i in range(10):
        func()
        time.sleep(0.1)

        print(1/FPSCalculator.get_execution_time('func', duration=5))
