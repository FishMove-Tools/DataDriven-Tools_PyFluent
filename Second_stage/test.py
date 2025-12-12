"""Evaluate Phase-2 PPO policy with optional visualisation."""
import argparse
import os
from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from fish_env import FishdatadrivenEnvPhase2

# 设置与导入模型和归一化器的路径
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=Path, default="./models_phase2/fish_model_phase2_60000_steps.zip")
    p.add_argument("--vecnorm_path", type=Path, default="./models_phase2/vec_normalize_phase2.pkl")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--render_mode", choices=["human", "rgb_array", "none"], default="human")
    return p.parse_args()

# 主函数
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 导入动力学模型
    def make_env():
        return Monitor(
            FishdatadrivenEnvPhase2(
                model_path_increase="fish_dynamics_increase.pth",
                model_path_decrease="fish_dynamics_decrease.pth",
                model_path_hengding="fish_dynamics_single.pth",
                max_steps=300,
                fixed_steps=100,
                render_mode=None if args.render_mode == "none" else args.render_mode,
            )
        )
    # 归一化
    vec_env = DummyVecEnv([make_env])

    if args.vecnorm_path.exists():
        vec_env = VecNormalize.load(args.vecnorm_path, vec_env)
        vec_env.training, vec_env.norm_reward = False, False
        print("VecNormalize statistics loaded.")
    else:
        print("VecNormalize statistics NOT found.")
    # 加载ppo模型
    model = PPO.load(args.model_path, env=vec_env, device=device)
    print(f"Loaded model on {device}.")

    env_raw = vec_env.venv.envs[0].env 
    # 计算奖励与打印
    for ep in range(1, args.episodes + 1):
        obs, done, ep_ret, step_idx = vec_env.reset(), False, 0.0, 0
        print(f"\n=== Episode {ep} ===")
        while not done:
            x, v, a = env_raw.state[:3]
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, infos = vec_env.step(action)

            r = reward[0]
            info = infos[0]
            done = done_arr[0]
            w, a1 = info["action_w"], info["action_a1"]
            step_idx += 1

            print(
                f"step {step_idx:04d} | x={x:7.3f}, v={v:7.3f}, a={a:7.3f} | "
                f"w={w:5.3f}, a1={a1:4.2f} | reward={r:8.2f}"
            )
            ep_ret += r

        print(f"Episode {ep} finished. Total reward = {ep_ret:.2f}")

    vec_env.close()
    print("\nEvaluation completed.")

if __name__ == "__main__":
    main()
