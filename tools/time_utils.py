import time

class TimeUtil:

    # return unix milliseconds
    @staticmethod
    def get_time():
        return int(time.time()*1000)