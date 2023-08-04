import time
class time_tracker():
    time = 0.
    begin = 0.
    end = 0.
    def get_cost_time():
        return time_tracker.time
    def tracker_begin():
        time_tracker.begin = time.time()
    def tracker_end():
        time_tracker.end = time.time()
        time_tracker.time += time_tracker.end - time_tracker.begin