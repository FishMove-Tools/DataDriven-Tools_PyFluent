import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import random
import ansys.fluent.core as pyfluent
import time
import math
import csv
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

class FluentEnv(gym.Env):
    """Fluent Environment for CFD simulations."""
    
    def __init__(self, max_steps=300, reward_function='swim_switch', simu_name="CFDSimulation"):
        super().__init__()
        
        print("--- Fluent CFD env init ---")
        # 定义动作空间和观测空间
        self.w_options = [3.14, 3.77, 4.398, 5.027, 5.655]
        self.a1_options = [0.02]
        self.delta_options = [-1, 0, 1]
        self.action_space = spaces.Discrete(len(self.delta_options))        
        # 初始化参数
        self.reward_function = reward_function
        self.simu_name = simu_name
        self.ncrash = 0      
        self.rewardd = 0.0
        self.x = 0
        self.ve = 0 
        self.a = 0.0
        self.last_w_index = 0
        self.last_w = self.w_options[self.last_w_index]
        self.substeps_left = 0.0        
        self.cut_judge = 0.0
        self.max_steps = max_steps 
        self.episode_number = 0                
        
        
        # 定义观测空间
        self.observation_space = spaces.Box(
            low=np.array([-20.0, -1.0, -1.0, 0.0, 0.0]),
            high=np.array([20.0, 1.0, 1.0, float(len(self.w_options) - 1), 400]),
            dtype=np.float64
        )
        
        # 环境状态 
        self.state = None
        # 频率到子步数的映射（与数据驱动环境完全一致）
        self.w2steps = {
            3.14: 400,
            3.77: 334,
            4.398: 286,
            5.027: 250,
            5.655: 222,
            6.28: 200
        }             
        # 日志文件初始化
        self.log_file = "training_log.csv"   
            
        # 初始化 Fluent
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
        print("--- Fluent CFD env init done! ---")
            
    def start_class(self, complete_reset=True):
        self.episode_number = 0
        self.action_number = 0
        self.current_step = 0        
        self.previous_modes_number = 0      
        self.ready_to_use = True
        self.initialize_flow(complete_reset=True)

    def close(self):
        self.ready_to_use = False
        if hasattr(self, 'solver'):
            self.solver.exit()

    def seed(self, seed=None):
        return seed

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.initialize_flow(complete_reset=True)
        self.x = 0.0
        self.ve = 0.0
        self.a = 0.0
        self.current_step = 0
        self.current_action_step = 0
        self.last_w_index = 0
        self.last_w = self.w_options[self.last_w_index]
        self.w_index = self.last_w_index     # ← 保证属性同步
        self.cut_judge = 0.0
        self.episode_number += 1

        # ★ 先写好 state
        self.state = np.array(
            [self.x, self.ve, self.a,
            self.last_w_index,
            self.current_action_step / 10],
            dtype=np.float64
        )

        # 再返回观测
        return self._get_obs(), {}


    def initialize_flow(self, complete_reset=True):
        if complete_reset:
            print(os.getcwd())
            print('reset the CFD environment')
            if self.episode_number > 1:
                self.solver.file.read_case_data(file_name="initial.cas.h5")
                self.scaled_w = 3.14
                self.scaled_a1 = 0.02      

    def custom_save(self, df, filename):
        with open(filename, 'w') as f:
            f.write(' '.join(df.columns) + '\n')
            for row in df.itertuples(index=False, name=None):
                f.write(f"{int(row[0])} {row[1]:.8e} {row[2]:.8e} {row[3]:.8e}\n")
                
    def _get_obs(self):
        if self.state is None:
            # 首次 reset 之前的极端情况
            return np.zeros(5, dtype=np.float64)
        return self.state.copy()

        
    def step(self, action):
        self.solver.solution.run_calculation.transient_controls.time_step_size = 0.005              
            # 边界处理
        delta = self.delta_options[action]
        current_w_index = self.last_w_index

        # 更新频率索引与动作值
        new_w_index = np.clip(current_w_index + delta, 0, len(self.w_options) - 1)
        self.scaled_w = self.w_options[new_w_index]
        self.scaled_a1 = self.a1_options[0]
        
        current_action_step = self.w2steps.get(self.scaled_w, 100)
        remaining_steps = self.max_steps - self.current_step            
        substeps_to_execute = min(current_action_step, remaining_steps)             
        for _ in range(substeps_to_execute):           
            # 执行Fluent计算
            u = float(self.scaled_w)
            v = float(self.scaled_a1)                                    
            self.solver.execute_tui(r'''(rpsetvar 'w1 {})'''.format(u))
            self.solver.execute_tui(r'''(rpsetvar 'a1 {})'''.format(v))                        
            self.solver.solution.run_calculation.dual_time_iterate(
                time_step_count=1,
                max_iter_per_step=40
            )                                                  
            self.current_step += 1
            self.x = np.loadtxt('positionx.txt')[-1,-3]
            self.ve = np.loadtxt('positionx.txt')[-1,-2]
            self.a = np.loadtxt('positionx.txt')[-1,-1] 
            self.state[0] = self.x
            self.state[1] = self.ve 
            self.state[2] = self.a               
        self.last_w_index = new_w_index
        self.last_w = self.scaled_w              
        self.state[3] = new_w_index
        self.state[4] = (current_action_step / 10)      
        reward = self._calculate_reward()
        if self.state[0] < -5.25:
            reward = -100.0
        if self.state[0] < -5.5:
            reward = -300.0
        if self.state[0] < -5.8:
            reward = -400.0 
        if self.state[0] < -6:
            reward = -500.0         
        self.rewardd = reward
         # 👇 合法终止条件：步满 or 超出 x
        done = (self.current_step >= self.max_steps) or (self.state[0] < -6)

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
        with open(filename, mode, encoding="utf-8") as csv_file:
            spam_writer = csv.writer(csv_file, delimiter=',', lineterminator="\n")
            spam_writer.writerow([
                self.state[0],   # x
                self.state[1],   # velocity
                self.state[2],   # accel
                self.scaled_w,
                self.scaled_a1,
            ])
            

    def close(self):
        pass            