import time

class ProgressBar:

    def __init__(self, max_number):
        self.max_number = max_number
        self.last_time = None
        self.last_number = None
        self.started_time = None
    
    def print_progress(self, current_number):
        # Compute duration of last iteration
        time_left = '?'
        current_time = time.time()
        if current_number > 0:
            time_since_start = current_time - self.started_time
            sec_per_i = time_since_start / current_number
            sec_left = (self.max_number - current_number) * sec_per_i
            time_left = time.strftime("%Hh%Mmin%Ss", time.gmtime(sec_left))
        else:
            self.started_time = current_time

        elapsed_time = time.strftime(
            "%Hh%Mmin%Ss", time.gmtime(current_time - self.started_time))

        percentage_done = 100 * current_number / self.max_number
        
        print(f"Step {current_number}/{self.max_number}, {percentage_done}% \
done, started {elapsed_time} ago, estimated time left: {time_left} seconds.", 
                end='\r')

        self.last_time = current_time
        self.last_number = current_number

    def print_end(self):
        elapsed_time = time.strftime(
            "%Hh%Mmin%Ss", time.gmtime(time.time() - self.started_time))
        print()
        print(f"Training ended after {elapsed_time}.")