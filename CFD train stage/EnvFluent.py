import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random
import ansys.fluent.core as pyfluent
import time
import math
import csv
import pandas as pd

class FluentEnv(gym.Env):
    """Final version of Fluent CFD Environment synchronized with data-driven Phase2."""
    def __init__(self, max_steps=300, reward_function='swim_switch', simu_name="CFDSimulation"):
        super().__init__()
        print("--- Fluent CFD env init ---")

        # Action and Observation Space
        self.w_options = [3.14, 3.77, 4.398, 5.027, 5.655]
        self.a1_options = [0.02]
        self.delta_options = [-1, 0, 1]
        self.action_space = spaces.Discrete(len(self.delta_options))

        self.observation_space = spaces.Box(
            low=np.array([-20.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([20.0, 1.0, 1.0, float(len(self.w_options) - 1), 80]),
            dtype=np.float64
        )

        self.w2steps = {
            3.14: 40, 3.77: 33, 4.398: 29,
            5.027: 25, 5.655: 22, 6.28: 20
        }

        # State Initialization
        self.max_steps = max_steps
        self.reward_function = reward_function
        self.simu_name = simu_name
        self.log_file = "training_log.csv"
        self.device = None

        self.episode_number = 0
        self.current_step = 0
        self.last_w_index = 4
        self.last_w = self.w_options[self.last_w_index]
        self.scaled_w = self.last_w
        self.scaled_a1 = self.a1_options[0]
        self.state = np.array([0.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float64)
        self.forced_action_sequence = None
        self.use_forced_actions = True
        self.max_forced_episodes = 5

        try:
            df_action = pd.read_excel("Action_policy.xlsx")
            self.forced_action_sequence = df_action.iloc[:, 0].dropna().astype(int).tolist()
            msg = f"{len(self.forced_action_sequence)}  Steps of forced actions loaded from Action_policy.xlsx."
            print(msg)
            
            # logging
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Failed to load Action_policy.xlsx: {e}")

        # Fluent Solver Initialization
        os.chdir('fishmove')
        self.solver = pyfluent.launch_fluent(
            product_version="23.1.0",
            precision="double",
            processor_count=4,
            mode="solver",
            version="2d",
            show_gui=True
        )
        self.solver.file.read_case_data(file_name="fishmove.cas.h5")
        self.start_class(complete_reset=True)
        print(" Fluent CFD env init done!")

    def start_class(self, complete_reset=True):
        self.episode_number = 0
        self.current_step = 0
        self.initialize_flow(complete_reset)

    def close(self):
        if hasattr(self, 'solver'):
            self.solver.exit()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.initialize_flow(complete_reset=True)
        self.x = 0.0
        self.ve = 0.0
        self.a = 0.0
        self.current_step = 0
        self.last_w_index = 4
        self.last_w = self.w_options[self.last_w_index]
        self.scaled_w = self.last_w
        self.scaled_a1 = self.a1_options[0]
        self.episode_number += 1
        if self.episode_number > self.max_forced_episodes:
            self.use_forced_actions = False
        self.state = self._get_obs()
        return self.state, {}

    def initialize_flow(self, complete_reset=True):
        if complete_reset:
            print("Resetting Fluent CFD case...")
            if self.episode_number > 0:
                self.solver.file.read_case_data(file_name="initial.cas.h5")
            self.scaled_w = 5.655
            self.scaled_a1 = 0.02

    def _get_obs(self):
        return np.array([0.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float64)

    def step(self, action):
        self.solver.solution.run_calculation.transient_controls.time_step_size = 0.005
        if self.use_forced_actions and self.forced_action_sequence is not None:
            if self.current_step < len(self.forced_action_sequence):
                action = int(self.forced_action_sequence[self.current_step])
                print(f"[Ep {self.episode_number}] delta={action} (step={self.current_step})")
            else:
                print(f"[Ep {self.episode_number}] No forced action for step {self.current_step}, using agent action.")
                
        # Begin action execution
        if self.current_step < 150:
            delta = 0
            new_w_index = 4  
        else:
            delta = self.delta_options[action]
            current_w_index = self.last_w_index
            if current_w_index == 0 and delta == -1:
                delta = 0
            elif current_w_index == len(self.w_options) - 1 and delta == 1:
                delta = 0
            new_w_index = np.clip(current_w_index + delta, 0, len(self.w_options) - 1)

        self.scaled_w = self.w_options[new_w_index]
        self.scaled_a1 = self.a1_options[0]

        current_action_step = self.w2steps.get(self.scaled_w, 10)
        remaining_steps = self.max_steps - self.current_step
        substeps_to_execute = min(current_action_step, remaining_steps)

        for _ in range(substeps_to_execute):
            u = float(self.scaled_w)
            v = float(self.scaled_a1)
            self.solver.execute_tui(f"""(rpsetvar 'w1 {u})""")
            self.solver.execute_tui(f"""(rpsetvar 'a1 {v})""")
            self.solver.solution.run_calculation.dual_time_iterate(time_step_count=10, max_iter_per_step=40)
            self.current_step += 1
            
            # Read position data
            self.x = np.loadtxt('positionx.txt')[-1, -3]
            self.ve = np.loadtxt('positionx.txt')[-1, -2]
            self.a = np.loadtxt('positionx.txt')[-1, -1]

        # Update state
        self.last_w_index = new_w_index
        self.last_w = self.scaled_w
        self.state[0] = self.x
        self.state[1] = self.ve
        self.state[2] = self.a
        self.state[3] = new_w_index
        self.state[4] = current_action_step

        reward = self._calculate_reward()

        if self.x < -5.25: reward = -100.0
        if self.x < -5.5:  reward = -300.0
        if self.x < -5.8:  reward = -400.0
        if self.x < -6.0:  reward = -500.0
        done = (self.current_step >= self.max_steps) or (self.x < -6)
        
        # If done, read final state from file
        if done:
            try:
                self.x = np.loadtxt('positionx.txt')[-1, -3]
                self.ve = np.loadtxt('positionx.txt')[-1, -2]
                self.a = np.loadtxt('positionx.txt')[-1, -1]

                self.state[0] = self.x
                self.state[1] = self.ve
                self.state[2] = self.a
            except Exception as e:
                print(f"Can not get the correct final state: {e}")        
        info = {
            "current_action_step": substeps_to_execute,
            "action_w": self.scaled_w,
            "action_a1": self.scaled_a1
        }
        return self.state.copy(), reward, done, False, info

    def _calculate_reward(self):
        distance = abs(self.x + 5.0)
        reward = -10 * distance
        if distance < 0.25:
            reward += 1000
        elif distance < 0.5:
            reward += 800
        return reward

    def _log_variables(self):
        filename = "varable_record.txt"
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode, encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',', lineterminator="\n")
            writer.writerow([
                self.state[0], self.state[1], self.state[2],
                self.scaled_w, self.scaled_a1
            ])