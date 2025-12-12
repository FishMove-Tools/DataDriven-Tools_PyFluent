import os
import csv
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import custom environment
# Ensure EnvFluent is in your PYTHONPATH
from EnvFluent import FluentEnv

# Setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL_PATH = "./saved_models/fish_final_phase2.zip"
VECNORM_PATH = "./saved_models/vec_normalize_phase2.pkl"
SAVE_PATH = "./saved_models"
LOG_PATH = "./logs"

class EpisodeCheckpointCallback(BaseCallback):
    """
    Custom callback for saving episode rewards, detailed action logs, 
    and model checkpoints based on performance.
    """
    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0

        # Setup logging paths
        self.reward_log_path = os.path.join(save_path, "episode_rewards.csv")
        self.action_log_folder = os.path.join(save_path, "action_logs")
        os.makedirs(self.action_log_folder, exist_ok=True)

        # Initialize global reward log file
        with open(self.reward_log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Episode", "Reward", "Mean_Reward", "Timesteps"])

        # Per-episode logging variables
        self.step_counter = 0
        self.episode_file = None
        self.episode_writer = None

    def _get_physical_state(self, obs, env):
        """
        Helper to extract physical state (x, v, a) from observations.
        Attempts to fetch un-normalized data if the environment supports it.
        """
        # Default: use current (possibly normalized) observations
        x, v, a = obs[0], obs[1], obs[2]
        
        # Try to retrieve original, un-normalized observations
        if hasattr(env, "get_original_obs"):
            try:
                # Assuming get_original_obs returns a list/array where index 0 is the obs
                original_obs = env.get_original_obs()[0]
                x, v, a = original_obs[0], original_obs[1], original_obs[2]
            except Exception as e:
                # Only print this error once per episode to avoid spamming, or use logging
                if self.step_counter == 1: 
                    print(f"Warning: Could not extract original observations, using normalized values. Reason: {e}")
        return x, v, a

    def _on_step(self) -> bool:
        # Retrieve local variables from the PPO learn function
        reward = self.locals['rewards'][0]
        obs = self.locals['new_obs'][0]
        done = self.locals['dones'][0]
        action = self.locals['actions'][0]
        info = self.locals['infos'][0]

        # Prevent logging the initial state immediately after a reset in the middle of a rollout
        if self.step_counter > 0 and done and self.step_counter < self.training_env.get_attr("max_steps")[0]:
            return True

        # Extract Action Info
        w = info.get("action_w", -1)
        # a1 = info.get("action_a1", -1) # Uncomment if needed
        w_index = int(obs[3]) if len(obs) > 3 else -1
        delta = int(action)

        # Initialize CSV writer for the new episode
        if self.step_counter == 0:
            episode_filename = os.path.join(self.action_log_folder, f"episode_{self.episode_count + 1}_actions.csv")
            self.episode_file = open(episode_filename, 'w', newline='', buffering=1)
            self.episode_writer = csv.DictWriter(self.episode_file, fieldnames=[
                "step", "x", "v", "a", "w", "w_index", "delta", "reward", "done"
            ])
            self.episode_writer.writeheader()

        self.step_counter += 1
        
        # Get physical state (handling normalization)
        x, v, a = self._get_physical_state(obs, self.training_env)

        # Log current step
        if self.episode_writer:
            self.episode_writer.writerow({
                "step": self.step_counter,
                "x": x, "v": v, "a": a,
                "w": w, "w_index": w_index,
                "delta": delta,
                "reward": reward,
                "done": done
            })
            self.episode_file.flush()

        self.current_episode_reward += reward

        # === Handle Episode End ===
        if done:
            try:
                # Log the final frame (post-termination state)
                final_obs = self.locals['new_obs'][0]
                final_reward = self.locals['rewards'][0]
                final_done = self.locals['dones'][0]
                final_action = self.locals['actions'][0]
                final_info = self.locals['infos'][0]

                final_w = final_info.get("action_w", -1)
                final_w_index = int(final_obs[3]) if len(final_obs) > 3 else -1
                final_delta = int(final_action)
                
                final_x, final_v, final_a = self._get_physical_state(final_obs, self.training_env)

                if self.episode_writer:
                    self.episode_writer.writerow({
                        "step": self.step_counter + 1,
                        "x": final_x, "v": final_v, "a": final_a,
                        "w": final_w, "w_index": final_w_index,
                        "delta": final_delta,
                        "reward": final_reward,
                        "done": final_done
                    })
                    self.episode_file.flush()
            except Exception as e:
                print(f"Warning: Failed to log the final frame: {e}")

            # Update stats and logs
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            
            # Calculate mean reward over last 10 episodes
            mean_reward = sum(self.episode_rewards[-10:]) / len(self.episode_rewards[-10:])
            
            # Write to global log
            with open(self.reward_log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.episode_count, 
                    self.current_episode_reward, 
                    mean_reward, 
                    self.num_timesteps
                ])
            
            print(f"[Ep {self.episode_count:03d}] Reward: {self.current_episode_reward:.2f} | Mean10: {mean_reward:.2f}")

            # Save checkpoint
            self.model.save(os.path.join(self.save_path, f"checkpoint_{self.episode_count}"))

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))

            # Close file handles and reset counters
            if self.episode_file:
                self.episode_file.close()
                self.episode_file = None
                self.episode_writer = None
            
            self.step_counter = 0
            self.current_episode_reward = 0

        return True

def add_noise_to_policy(model, noise_std=0.01):
    """
    Adds small Gaussian noise to the policy network parameters 
    to break local optima and encourage exploration.
    """
    with torch.no_grad():
        for param in model.policy.parameters():
            noise = torch.randn_like(param) * noise_std
            param.data += noise
    print("Added small perturbations to policy parameters to enhance exploration.")

def make_env():
    """Create and monitor the environment."""
    return Monitor(FluentEnv(max_steps=300), LOG_PATH)

def train_ppo_model(total_timesteps=9000):
    # Ensure directories exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    # Initialize Environment
    env = DummyVecEnv([make_env])
    
    # Handle Observation Normalization
    if os.path.exists(VECNORM_PATH):
        print(f"Loading VecNormalize statistics from {VECNORM_PATH}")
        env = VecNormalize.load(VECNORM_PATH, env)
    else:
        print("Initializing new VecNormalize wrapper.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Network Architecture
    policy_kwargs = dict(net_arch=dict(pi=[512, 512], vf=[128, 128]))

    # Load Model or Train from Scratch
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading Pretrained PPO model from {PRETRAINED_MODEL_PATH}")
        model = PPO.load(PRETRAINED_MODEL_PATH, env=env, device=DEVICE)
    else:
        print("Training from scratch...")
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    learning_rate=1e-3, n_steps=2048, batch_size=64, n_epochs=10,
                    gamma=0.99, gae_lambda=0.97, clip_range=0.2, ent_coef=0.01,
                    verbose=1, tensorboard_log=LOG_PATH, device=DEVICE)

    # Apply noise to weights (Exploration strategy)
    add_noise_to_policy(model, noise_std=0.01)

    # Setup Callbacks
    checkpoint_cb = CheckpointCallback(save_freq=300, save_path=SAVE_PATH, name_prefix="checkpoint")
    episode_cb = EpisodeCheckpointCallback(save_path=SAVE_PATH, verbose=1)
    callback = CallbackList([checkpoint_cb, episode_cb])

    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=callback,
                    reset_num_timesteps=False)

        # Save Final Model and Env Stats
        model.save(os.path.join(SAVE_PATH, "final_cfd_model"))
        env.save(os.path.join(SAVE_PATH, "vec_normalize_cfd.pkl"))
        print("Training finished and models saved.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving backup...")
        model.save(os.path.join(SAVE_PATH, "interrupted_cfd_model"))
        env.save(os.path.join(SAVE_PATH, "vec_normalize_cfd_interrupted.pkl"))
    
    finally:
        env.close()

if __name__ == "__main__":
    train_ppo_model(total_timesteps=9000)