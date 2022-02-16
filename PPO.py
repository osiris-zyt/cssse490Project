# %%
from tkinter import N
import numpy as np
import gym
import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from collections import deque
from torch.autograd import Variable
import random
import math

import pandas as pd
from tqdm.std import tqdm
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
env = gym.make('Pendulum-v0')
n_state = int(np.prod(env.observation_space.shape))
n_action = int(np.prod(env.action_space.shape))
print("# of state", n_state)
print("# of action", n_action)

# %%
# -------------------------------------------- ************** --------------------------------------------

def run_episode(env, policy, render=False):
    obs_list = []
    act_list = []
    reward_list = []
    next_obs_list = []
    done_list = []
    obs = env.reset()
    while True:
        if render:
            env.render()

        action, mu, var = policy(obs)
        next_obs, reward, done, _ = env.step(action)
        reward_list.append(reward), obs_list.append(obs), \
            done_list.append(done), act_list.append(action), \
            next_obs_list.append(next_obs)
        if done:
            break
        obs = next_obs

    return obs_list, act_list, reward_list, next_obs_list, done_list

# %%

# -------------------------------------------- ************** --------------------------------------------

class MBMPO():
    def __init__(self, n_state, n_action, n_model = 3):

        # N-Model
        self.models = []
        self.model_optimzier = []
        for _ in range(n_model):
            model = nn.Sequential(
                nn.Linear(n_state + n_action, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, n_state)
            )
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            self.model_optimzier.append(optimizer)
            self.models.append(model)

        # Meta-Policy
        self.meta_policy = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*n_action),
        )
        self.meta_policy.to(device)
        self.old_act_net = copy.deepcopy(self.meta_policy)
        self.old_act_net.to(device)

        # Sub-Policy
        self.sub_policies = []
        self.sub_optimizers = []
        for _ in range(n_model):
            policy = nn.Sequential(
                nn.Linear(n_state, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 2*n_action),
            )
            optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
            policy.to(device)
            self.sub_policies.append(policy)
            self.sub_optimizers.append(optimizer)

        # V Net
        self.v_net = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.v_net.to(device)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=1e-3)
        self.meta_act_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(), lr=1e-3)
        self.old_v_net = copy.deepcopy(self.v_net)
        self.old_v_net.to(device)
        self.gamma = 0.95
        self.alpha = 0.95
        self.beta = 0.95
        self.gae_lambda = 0.85
        self._eps_clip = 0.2
        self.act_lim = 2

        # Data store
        self.d = ReplayBuffer(50000)

    ## Call Section ----------------------------------------------------------

    def __call__(self, state, index = -1):
        # -1 is meta policy. 0,1,2,3... are sub_policies
        with torch.no_grad():
            policy = self.meta_policy if index == -1 else self.sub_policies[index]
            state = torch.FloatTensor(state).to(device)
            # calculate act prob
            output = policy(state)
            mu = self.act_lim*torch.tanh(output[:n_action])
            var = torch.abs(output[n_action:])
            dist = Normal(mu, var)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim), mu, var

    def next_act(self, index, state):
            # -1 is meta policy. 0,1,2,3... are sub_policies
        with torch.no_grad():
            policy = self.meta_act_net if index == -1 else self.sub_policies[index]
            state = torch.FloatTensor(state).to(device)
            # calculate act prob
            #print(policy.parameters())

            output = policy(state)
            mu = self.act_lim*torch.tanh(output[:n_action])
            var = torch.abs(output[n_action:])
            dist = Normal(mu, var)
            action = dist.sample()
            action = action.detach().cpu().numpy()
        return np.clip(action, -self.act_lim, self.act_lim), mu, var


    ## Sampling Section ----------------------------------------------------------

    def sampleFromEnv(self, index):
        obs = env.reset()
        while True:
            act, mu, var = self.next_act(index, obs)
            next_obs, reward, done, _ = env.step(act)
            next_obs = next_obs.squeeze()
            self.d.add(obs, act, mu, var, reward, next_obs, done) # ?? var and mu
            obs = next_obs
            if done:
                break

        return
         

    def sampleNStep(self, policy, model, step = 20):
        # Similar to Run Episode
        obs_list = []
        act_list = []
        mu_list = []
        var_list = []
        reward_list = []
        next_obs_list = []
        done_list = []
        obs = env.reset()
        obs = torch.FloatTensor(obs).to(device)
        with torch.no_grad():
            for i in range(0,step):
                output = policy(obs)
                mu = self.act_lim*torch.tanh(output[:n_action])
                var = torch.abs(output[n_action:])
                dist = Normal(mu, var)
                action = dist.sample().reshape(-1)
                model_input = torch.cat([obs,action]) 
                next_obs = obs + model(model_input)
                reward = reward_calc(obs, action)
                reward_list.append(reward.detach().numpy()), obs_list.append(obs.detach().numpy()), \
                    mu_list.append(mu.detach().numpy()), var_list.append(var.detach().numpy()), act_list.append(action.detach().numpy()), \
                    next_obs_list.append(next_obs.detach().numpy())
                    
                obs = next_obs

        return np.array(obs_list), np.array(act_list), np.array(mu_list), np.array(var_list), \
                    np.array(reward_list), np.array(next_obs_list)
        




        #     reward_list.append(reward), obs_list.append(obs), \
        #         mu_list.append(mu), var_list.append(var), act_list.append(action), \
        #         next_obs_list.append(next_obs)
                
        #     obs = next_obs

        # return obs_list, act_list,mu_list, var_list, reward_list, next_obs_list

    ## Update Section ----------------------------------------------------------

    def updateModel(self, index):
        # TODO: Sample from replay buffer
        model = self.models[index]
        optimizer = self.model_optimzier[index]
        obs, act, mu, var, reward, next_obs, done = self.d.sample(32)
        diff = next_obs - obs
        model_input = torch.cat([obs, act], axis = 1)
        diff_h = model(model_input)
        loss = F.mse_loss(diff, diff_h)
        loss.backward()
        optimizer.step()

        return

    def updateSubPolicy(self, index, data=None):
        ## Update using batch or one step by one step along the trajectory?
        policy =  self.sub_policies[index]
        optimizer = self.sub_optimizers[index]
        # Policy gradient
        obs, act, mu, var, reward, next_obs = data # ?? How to use mu and var here
        mu = torch.FloatTensor(mu)
        var = torch.FloatTensor(var)
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)
        reward = torch.FloatTensor(reward)
        next_obs = torch.FloatTensor(next_obs)
        # Calculate culmulative return
        returns = np.zeros(len(reward))
        s = 0
        for i in reversed(range(len(returns))):
            s = s * self.gamma + reward[i]
            returns[i] = s

        returns = torch.FloatTensor(returns)
        # Calculate loss
        batch_size = 32
        list = [j for j in range(len(obs))]
        for i in range(0, len(list), batch_size):
            ## ?? Use the mu and var from the subpolicy instead of from the sampling?
            index = list[i:i+batch_size]
            output = policy(obs[index, :])
            mu = self.act_lim*torch.tanh(output[:,:n_action])
            var = torch.abs(output[:,n_action:])
            dist = Normal(mu, var)
            logprob = dist.log_prob(act[index]).squeeze_() ## ?????
            # logprob = torch.log(policy(obs[index, :])[0])
            gt_logprob = returns[index] * logprob
            loss = -gt_logprob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    def updateMetaPolicy(self, data=None):
        # PPO
        done = 0
        obs, act, mu, var, reward, next_obs = data
        obs = torch.FloatTensor(obs).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        act = torch.FloatTensor(act).to(device)
        mu = torch.FloatTensor(mu).to(device)
        var = torch.FloatTensor(var).to(device)
        with torch.no_grad():
            v_s = self.old_v_net(obs).detach().cpu().numpy().squeeze()
            v_s_ = self.old_v_net(next_obs).detach().cpu().numpy().squeeze()
            # calculate the pi_theta_k from current policy

            ## TODO: Change to sub-policy
            #output = self.old_act_net(obs)
            #mu = self.act_lim*torch.tanh(output[:, :n_action])
            #var = torch.abs(output[:, n_action:])
            dist = Normal(mu, var)
            old_logprob = dist.log_prob(act)

        adv = np.zeros_like(reward)
        done = np.array(done, dtype=float)

        returns = np.zeros_like(reward)
        # # One-step
        # adv = reward + (1-done)*self.gamma*v_s_ - v_s
        # returns = adv + v_s
        # MC
        # s = 0
        # for i in reversed(range(len(returns))):
        #     s = s * self.gamma + reward[i]
        #     returns[i] = s
        # adv = returns - v_s
        # # GAE
        reward = reward.squeeze()
        delta = reward + v_s_ * self.gamma - v_s
        m = (1.0 - done) * (self.gamma * self.gae_lambda)
        gae = 0.0
        for i in range(len(reward) - 1, -1, -1):
            gae = delta[i] + m * gae #?????? (m[i])
            adv[i] = gae
        returns = adv + v_s

        adv = torch.FloatTensor(adv).to(device)
        returns = torch.FloatTensor(returns).to(device)
        # Calculate loss
        batch_size = 32
        list = [j for j in range(len(obs))]
        for i in range(0, len(list), batch_size):
            index = list[i:i+batch_size]
            for _ in range(1):
                output = self.meta_policy(obs[index])
                mu = self.act_lim*torch.tanh(output[:, :n_action])
                var = torch.abs(output[:, n_action:])
                dist = Normal(mu, var)
                logprob = dist.log_prob(act[index])
                ## TODO: Old log should be replaced by log of sub-policy
                ratio = (logprob - old_logprob[index]).exp().float().squeeze()
                surr1 = ratio * adv[index]
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 +
                                    self._eps_clip) * adv[index]
                act_loss = -torch.min(surr1, surr2).mean()

                ent_loss = dist.entropy().mean()
                act_loss -= 0.01*ent_loss
                self.meta_act_optimizer.zero_grad()
                act_loss.backward()
                self.meta_act_optimizer.step()

            for _ in range(1):
                v_loss = F.mse_loss(self.v_net(
                    obs[index]).squeeze(), returns[index])
                self.v_optimizer.zero_grad()
                v_loss.backward()
                self.v_optimizer.step()

        return act_loss.item(), v_loss.item(), ent_loss.item()

# -------------------------------------------- ************** --------------------------------------------

#Normalize function for calculating reward
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

#Reward Function
def reward_calc(state, action):
    theta_dot = state[2]
    theta = math.atan2(state[1], state[0])
    costs = angle_normalize(theta) ** 2 + 0.1 * theta_dot ** 2 + 0.001 * (action ** 2)
    return -costs

class ReplayBuffer:
    def __init__(self, size):
        self.buff = deque(maxlen=size)

    def add(self, obs, act, mu, var, reward, next_obs, done):
        self.buff.append([obs, act, mu, var, reward, next_obs, done])

    # def sample(self): # Pop the last record ??????????
    #     if(len(self.buff) < 1):
    #         return

    #     sample = self.buff.pop()
    #     obs = torch.FloatTensor([exp[0] for exp in sample]).to(device)
    #     act = torch.FloatTensor([exp[1] for exp in sample]).to(device)
    #     reward = torch.FloatTensor([exp[2] for exp in sample]).to(device)
    #     next_obs = torch.FloatTensor([exp[3] for exp in sample]).to(device)
    #     done = torch.FloatTensor([exp[4] for exp in sample]).to(device)
    #     return obs, act, reward, next_obs, done


    def sample(self, sample_size):
        if(len(self.buff) < sample_size):
            sample_size = len(self.buff)

        sample = random.sample(self.buff, sample_size)
        obs = torch.FloatTensor([exp[0] for exp in sample]).to(device)
        act = torch.FloatTensor([exp[1] for exp in sample]).to(device)
        mu = torch.FloatTensor([exp[2] for exp in sample]).to(device)
        var = torch.FloatTensor([exp[3] for exp in sample]).to(device)
        reward = torch.FloatTensor([exp[4] for exp in sample]).to(device)
        next_obs = torch.FloatTensor([exp[5] for exp in sample]).to(device)
        done = torch.FloatTensor([exp[6] for exp in sample]).to(device)
        return obs, act, mu, var, reward, next_obs, done


    def __len__(self):
        return len(self.buff)






# %%
# Begin Experiment
# -------------------------------------------- ************** --------------------------------------------
n_model = 3
loss_act_list, loss_v_list, loss_ent_list, reward_list = [], [], [], []
mbmpo = MBMPO(n_state, n_action, n_model = n_model)
loss_act, loss_v = 0, 0
n_step = 0
for i in tqdm(range(2000)):
    for index in range(n_model):
        mbmpo.sampleFromEnv(index) # Step 3

    for index in range(n_model):
        mbmpo.updateModel(index) # Step 4
        
    for index in range(n_model):
        policy = mbmpo.sub_policies[index]
        policy.load_state_dict(mbmpo.meta_policy.state_dict())
        optimizer = mbmpo.sub_optimizers[index]
        model = mbmpo.models[index]
        data = mbmpo.sampleNStep(policy, model, step = 20) # Step 6
        mbmpo.updateSubPolicy(index, data) # Step 7
        data = mbmpo.sampleNStep(policy, model, step = 20) # Step 8
        mbmpo.updateMetaPolicy(data) # Step 10
    ##End for (step 9)

    if i % 2 == 0:
        mbmpo.old_v_net.load_state_dict(mbmpo.v_net.state_dict())

    
    if i > 0 and i % 50 == 0:
        print(sum(run_episode(env, mbmpo, False)[2]))
        # print("itr:({:>5d}) loss_act:{:>6.4f} loss_v:{:>6.4f} loss_ent:{:>6.4f} reward:{:>3.1f}".format(i, np.mean(
        #     loss_act_list[-50:]), np.mean(loss_v_list[-50:]),
        #     np.mean(loss_ent_list[-50:]), np.mean(reward_list[-50:])))

    # loss_act_list.append(loss_act), loss_v_list.append(
    #     loss_v), loss_ent_list.append(loss_ent), reward_list.append(rew)


# %%
# Plot
# -------------------------------------------- ************** --------------------------------------------

scores = [sum(run_episode(env, mbmpo, False)[2]) for _ in range(100)]
print("Final score:", np.mean(scores))


df = pd.DataFrame({'loss_v': loss_v_list,
                   'loss_act': loss_act_list,
                   'loss_ent': loss_ent_list,
                   'reward': reward_list})
df.to_csv("./ClassMaterials/Lecture_25_PPO/data/ppo.csv",
          index=False, header=True)
