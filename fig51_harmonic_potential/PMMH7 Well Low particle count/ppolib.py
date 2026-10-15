from numba import njit, prange
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm.notebook import tqdm
from itertools import zip_longest
from ppolib import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pickle
import os
import shutil

'''
This uses major parts of CleanRL's PPO implementation
https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

@article{huang2022cleanrl,
  author  = {Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga and Dipam Chakraborty and Kinal Mehta and João G.M. Araújo},
  title   = {CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {274},
  pages   = {1--18},
  url     = {http://jmlr.org/papers/v23/21-1342.html}
}
'''


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ppo_network(nn.Module):
    def __init__(self, obs_space_size, action_space_size, hidden_layer_size=16):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_space_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, hidden_layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_layer_size, action_space_size), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space_size))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action_determenistic(self, x):
        action_mean = self.actor_mean(x)
        return action_mean


class PPO_Agent:

    def __init__(self, obs_space_size, action_space_size,
                 learning_rate=3e-4, hidden_layer_size=16,
                 tensorboard_suffix=None,
                 load_model=None,
                 name=None,
                ):
        self.obs_space_size = obs_space_size
        self.action_space_size = action_space_size

        if load_model:
            self.load_model(load_model)
        else:
            self.agent = ppo_network(obs_space_size, action_space_size, hidden_layer_size)
        self.recording = False
        self.awaiting_reward = False
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.norm_adv = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.learning_rate = learning_rate
        self.max_grad_norm = 0.5
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not name:
            self.name = timestamp
        else:
            self.name = name
        self.log_dir = f"runs/{self.name}"
        if tensorboard_suffix:
            self.log_dir += f"_{tensorboard_suffix}"

            
        self.episode_count = 0
        self.episode_reward = 0.
        self.episode_length = 0

        self.writer = None


    def start_tensorboard_writer(self):
        if not self.writer:
            self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def stop_tensorboard_writer(self):

        assert self.writer is not None, "No active writer"
        self.writer.close()
        self.writer = None
    
    def start_recording(self):
        assert not self.recording, "Can't start recording, already active"
        
        self.obs              = []
        self.actions          = []
        self.logprobs         = []
        self.rewards          = []
        self.fresh_episode    = []
        self.values           = []

        self.recording = True
        self.awaiting_reward = False
        

    def stop_recording(self):
        assert self.recording, "No active recording there to stop."
        assert not self.awaiting_reward, "Missing the last reward"
        self.recording = False
        pass

    def predict(self, obs, fresh_episode=None):

        assert (not self.recording) or (fresh_episode is not None), "Active recording requires fresh_episode flag"
        assert not self.awaiting_reward, "Awaiting reward for the last prediction"
        
        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(obs)

        self.episode_length += 1

        if self.recording:
            self.obs.append(obs)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.fresh_episode.append(fresh_episode)
            self.values.append(value.flatten())
            self.awaiting_reward = True

            if fresh_episode:
                if self.writer:
                    self.writer.add_scalar("Performance/Episode_Reward", self.episode_reward, self.episode_count)
                    self.writer.add_scalar("Performance/Episode_Reward_Quality", self.episode_reward / self.episode_length, self.episode_count)
                    self.writer.add_scalar("Performance/Episode_Length", self.episode_length, self.episode_count)
                    
                self.episode_reward = 0
                self.episode_count += 1
                self.episode_length = 0

        return action

    def predict_deterministic(self, obs):
        with torch.no_grad():
            action = self.agent.get_action_determenistic(obs)
        return action
        

    def reward(self, reward):
        assert self.awaiting_reward, "First need to predict before able to receive reward"
        assert self.recording, "No record active, can't reward"

        self.rewards.append(reward)
        self.awaiting_reward = False
        self.episode_reward += reward
        assert len(self.rewards) == len(self.actions)

    def print_tape(self):
        for o, a, lp, r, f, val in zip_longest(self.obs, self.actions,self.logprobs,
                                               self.rewards, self.fresh_episode,
                                               self.values, fillvalue="pending"):
            print(f"obs:{o}\taction:{a}\treward:{r}\tval:{val}")

    def scale_learning_rate(self, factor):
        self.optimizer.param_groups[0]["lr"] *= factor

    def report_telemetry(self, name, value):
        assert self.recording, "No recording active"
        assert self.writer, "No writer active"
        self.writer.add_scalar("Telemetry/" + name, value, self.episode_count)

    ''' -------------------- LOADING AND SAVING -------------------------- '''

    def save_model(self, filename=None):
        if filename is None:
            filename = f"agents/{self.name}"
    
        # Save the model weights
        torch.save(self.agent.state_dict(), filename + ".pth")
    
        # Save the rest of the PPO_Agent object state
        with open(filename + ".pkl", "wb") as file:
            pickle.dump(self, file)
        print(f"State saved to {filename}.")

    def load_model(self, filename=None):
        if filename is None:
            filename = f"agents/{self.name}"
    
        # Load the weights into a new agent instance
        self.agent.load_state_dict(torch.load(filename + ".pth"))

    @staticmethod
    def load(filename):
        # Load the PPO_Agent state
        with open(filename + ".pkl", "rb") as file:
            obj = pickle.load(file)
    
        # Load the model weights
        obj.agent.load_state_dict(torch.load(filename + ".pth"))
        print(f"State loaded from {filename}.")
        return obj

    def __getstate__(self):
        #if self.writer is not None:
        #    raise RuntimeError("Pickle prevented by the writer. .stop_tensorboard_writer() before saving the state")

        state = self.__dict__.copy()
        state["writer"] = None  # Exclude non-picklable attributes
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.writer = None  # Reinitialize manually

    def delete_tensorboard_data(self):
        if self.writer is not None:
            self.stop_tensorboard_writer()  # Stop the TensorBoard writer if it's active

        # Remove the TensorBoard log directory if it exists
        if os.path.exists(self.log_dir):
            try:
                shutil.rmtree(self.log_dir)  # Deletes the entire directory and its contents
                print(f"TensorBoard data at {self.log_dir} has been deleted.")
            except Exception as e:
                print(f"An error occurred while deleting TensorBoard data: {e}")
        else:
            print(f"No TensorBoard data found at {self.log_dir}.")


    ''' ---------------------------------------------------------------------- '''
        

    def learn(self, minibatch_size=32, epochs=1, debug=False):
        assert self.recording, "No recording active"
        assert not self.awaiting_reward, "Last reward still outstanding, can't learn"

        # Calculate returns using Generalized Advantage Estimation
        with torch.no_grad():
            n_records = len(self.obs)
            advantages = np.zeros(n_records)
            lastgaelam = 0

            for t in reversed(range(n_records)):
                #print(t)
                last_data_point = (t == n_records - 1)
                episode_continues = False if last_data_point else 1.0 - self.fresh_episode[t + 1]
                next_value = 0 if last_data_point else self.values[t + 1]
                this_value = self.values[t]

                delta = self.rewards[t] - this_value + next_value * self.gamma * episode_continues
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * episode_continues * lastgaelam

            #print(advantages)
            returns = T(advantages) + T(self.values)
            if debug:
                print("GAE returns", returns)

        # Make tensors
        b_obs = torch.concat(self.obs).reshape((-1,) + (self.obs_space_size,))
        b_logprobs = torch.concat(self.logprobs).reshape(-1)
        b_actions = torch.concat(self.actions).reshape((-1,) + (self.action_space_size,))
        b_advantages = T(advantages).reshape(-1)
        b_returns = T(returns).reshape(-1)
        b_values = torch.concat(self.values).reshape(-1)

        batch_size = len(self.obs)


        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                if mb_inds.shape[0] < minibatch_size:
                    if debug:
                        print("Skip incomplete minibatch")
                    continue

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if debug:
                    print(b_advantages)
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                if loss.isnan():
                    print("WIU WIU WIU")
                    print(locals())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.recording:
                    if self.writer:
                        self.writer.add_scalar("Telemetry/Policy_Loss", pg_loss.item(), self.episode_count)
                        self.writer.add_scalar("Telemetry/Value_Loss", v_loss.item(), self.episode_count)
                        self.writer.add_scalar("Telemetry/Entropy_Loss", entropy_loss.item(), self.episode_count)
                        self.writer.add_scalar("Telemetry/logLR", np.log10(self.optimizer.param_groups[0]["lr"]), self.episode_count)
                        self.writer.add_scalar("Telemetry/BatchSize", len(self.obs), self.episode_count)

        
