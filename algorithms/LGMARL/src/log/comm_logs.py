import os
import csv
import json


class CommunicationLogger:

    def __init__(self, save_dir):
        # Log dir
        self.log_dir = os.path.join(save_dir, "comm_logs")
        os.makedirs(self.log_dir)

        self.last_save_step = 0

    def log(self, policy_inputs, comm_rewards, comm_returns, perf_messages, 
            perf_broadcasts):
        # Flatten all data
        policy_inputs = policy_inputs[:-1].copy().reshape(
            -1, policy_inputs.shape[-1])
        comm_rewards = comm_rewards.copy().reshape(-1)
        comm_returns = comm_returns[:-1].copy().reshape(-1)
        perf_messages = [
            agent_message
            for step_messages in perf_messages[:-1]
            for env_messages in step_messages
            for agent_message in env_messages]
        perf_broadcasts = [
            agent_broadcast
            for step_broadcasts in perf_broadcasts[:-1]
            for env_broadcasts in step_broadcasts
            for agent_broadcast in env_broadcasts]

        # Log
        n_steps = len(perf_messages)
        save_path = os.path.join(
            self.log_dir, 
            f"cl{self.last_save_step}-{self.last_save_step + n_steps}.csv")

        with open(save_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                "Perfect Message", 
                "Broadcasted Message",
                "Comm Reward",
                "Comm Return",
                "Policy input"])
            for pm, br, rw, rt, pi in zip(
                    perf_messages, 
                    perf_broadcasts,
                    comm_rewards,
                    comm_returns,
                    policy_inputs):
                w.writerow([
                    " ".join(pm),
                    " ".join(br),
                    str(rw),
                    str(rt),
                    " ".join(str(pi_i) for pi_i in pi)])

        # Add number of steps to counter
        self.last_save_step = self.last_save_step + n_steps
