"""Proximal Policy Optimization (PPO) for learning bridge parameters (section 5.8.2).

Third-party code: this module is derived from ``cleanrl/ppo_continuous_action.py`` of CleanRL,
    https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
    (documentation: https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy)
    Copyright (c) 2019 CleanRL developers, MIT License; the license text is in LICENSES/CleanRL-MIT.txt.
Taken over: network architecture and initialisation (``layer_init``, ``PPONetwork``), hyperparameters, Generalized
Advantage Estimation and the clipped PPO update. Changed: the environment loop is replaced by a recording interface
(the particle filter calls the agent), no vectorised environments, no logging.
References: S. Huang et al., CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning
Algorithms, JMLR 23(274):1-18 (2022), http://jmlr.org/papers/v23/21-1342.html; J. Schulman et al., Proximal Policy
Optimization Algorithms, arXiv:1707.06347 (2017).

Actor and critic are fully connected networks with two hidden tanh layers; the policy is a Gaussian head
N(f_o(s), sigma_c) with a learned, state-independent width.

Usage pattern (one *recording* collects state-action-reward triplets over many episodes)::

    agent.start_recording()
    ...  agent.predict(obs, fresh_episode) / agent.reward(r)  ...
    agent.learn(minibatch_size=16, epochs=4)
    agent.stop_recording()
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPONetwork(nn.Module):
    def __init__(self, n_obs, n_act, hidden=16):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(n_obs, hidden)), nn.Tanh(),
                                    layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                    layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(n_obs, hidden)), nn.Tanh(),
                                        layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
                                        layer_init(nn.Linear(hidden, n_act), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act))

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        probs = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPOAgent:
    def __init__(self, n_obs, n_act, learning_rate=3e-4, hidden=16):
        self.n_obs, self.n_act = n_obs, n_act
        self.net = PPONetwork(n_obs, n_act, hidden)
        self.gamma, self.gae_lambda = 0.99, 0.95
        self.clip_coef, self.clip_vloss, self.norm_adv = 0.2, True, True
        self.ent_coef, self.vf_coef, self.max_grad_norm = 0.0, 0.5, 0.5
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, eps=1e-5)
        self.recording = self.awaiting_reward = False
        self.episode_count = 0
        self.episode_reward = 0.0
        self.episode_length = 0

    # ----------------------------------------------------------------- acting
    def start_recording(self):
        self.obs, self.actions, self.logprobs, self.rewards, self.fresh, self.values = [], [], [], [], [], []
        self.recording, self.awaiting_reward = True, False

    def stop_recording(self):
        self.recording = False

    def predict(self, obs, fresh_episode):
        with torch.no_grad():
            action, logprob, _, value = self.net.get_action_and_value(obs)
        self.episode_length += 1
        if self.recording:
            self.obs.append(obs); self.actions.append(action); self.logprobs.append(logprob)
            self.fresh.append(fresh_episode); self.values.append(value.flatten())
            self.awaiting_reward = True
            if fresh_episode:
                self.episode_reward, self.episode_length = 0.0, 0
                self.episode_count += 1
        return action

    def predict_deterministic(self, obs):
        with torch.no_grad():
            return self.net.actor_mean(obs)

    def reward(self, r):
        self.rewards.append(r)
        self.awaiting_reward = False
        self.episode_reward += r

    # --------------------------------------------------------------- learning
    def learn(self, minibatch_size=32, epochs=1):
        n = len(self.obs)
        with torch.no_grad():                                   # Generalized Advantage Estimation
            adv = np.zeros(n)
            last = 0.0
            for t in reversed(range(n)):
                cont = 0.0 if t == n - 1 else 1.0 - self.fresh[t + 1]
                nxt = 0.0 if t == n - 1 else self.values[t + 1]
                delta = self.rewards[t] - self.values[t] + nxt * self.gamma * cont
                adv[t] = last = delta + self.gamma * self.gae_lambda * cont * last
            returns = torch.Tensor(adv) + torch.Tensor(np.array([v.item() for v in self.values]))
        b_obs = torch.concat(self.obs).reshape(-1, self.n_obs)
        b_logp = torch.concat(self.logprobs).reshape(-1)
        b_act = torch.concat(self.actions).reshape(-1, self.n_act)
        b_adv = torch.Tensor(adv).reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = torch.concat(self.values).reshape(-1)
        idx = np.arange(n)
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n, minibatch_size):
                mb = idx[start:start + minibatch_size]
                if len(mb) < minibatch_size:
                    continue
                _, newlogp, entropy, newval = self.net.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                a = b_adv[mb]
                if self.norm_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)).mean()
                newval = newval.view(-1)
                if self.clip_vloss:
                    v_clip = b_val[mb] + torch.clamp(newval - b_val[mb], -self.clip_coef, self.clip_coef)
                    v_loss = 0.5 * torch.max((newval - b_ret[mb]) ** 2, (v_clip - b_ret[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()
                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def head_width(self):
        """Width sigma_c of the Gaussian head (per action dimension)."""
        return np.exp(self.net.actor_logstd.detach().numpy().ravel())
