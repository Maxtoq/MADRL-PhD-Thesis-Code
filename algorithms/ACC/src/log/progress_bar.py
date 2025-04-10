import time
import datetime


class Progress:

    def __init__(self, tot_steps):
        self.tot_steps = int(tot_steps)
        self.done_steps = 0
        self.start_time = time.time()

    def print_progress(self, step):
        elapsed_time = time.time() - self.start_time
        time_per_step = elapsed_time / (step + 1)

        steps_left = self.tot_steps - step
        time_left = steps_left * time_per_step

        progress_percent = min(int(100 * (step + 1) / self.tot_steps), 100)

        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        time_left_str = str(datetime.timedelta(seconds=int(time_left)))
        if elapsed_time >= step + 1:
            time_per_step = elapsed_time / (step + 1)
            tps_str = '%.2f'%(time_per_step) + "sec/step"
        else:
            step_per_sec = step / (elapsed_time + 1)
            tps_str = '%.2f'%(step_per_sec) + "step/sec"
        print(f"{progress_percent}% | {step}/{self.tot_steps} ({elapsed_time_str}<{time_left_str}, {tps_str})", end='\r')

    def print_end(self):
        elapsed_time = time.time() - self.start_time
        time_per_step = elapsed_time / self.tot_steps
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        if elapsed_time >= self.tot_steps:
            time_per_step = elapsed_time / self.tot_steps
            tps_str = '%.2f'%(time_per_step) + "sec/step"
        else:
            step_per_sec = self.tot_steps / elapsed_time
            tps_str = '%.2f'%(step_per_sec) + "step/sec"
        print(f"100% | {self.tot_steps}/{self.tot_steps} ({elapsed_time_str}, {tps_str})                    ")