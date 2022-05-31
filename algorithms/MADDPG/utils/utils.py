import time

class ProgressBar:

    def __init__(self, max_number):
        self.max_number = max_number
        self.last_time = None
        self.last_number = None
    
    def print_progress(self, current_number):
        # Compute duration of last iteration
        time_left = '?'
        current_time = time.time()
        if current_number > 0:
            dur = current_time - self.last_time
            sec_per_i = dur / (current_number - self.last_number)
            sec_left = (self.max_number - current_number) * sec_per_i
            time_left = time.strftime("%Hh%Mmin%Ss", time.gmtime(sec_left))
        
        print(f"Step {current_number}/{self.max_number}, estimated time left: {time_left} seconds.", 
                end='\r')

        self.last_time = current_time
        self.last_number = current_number