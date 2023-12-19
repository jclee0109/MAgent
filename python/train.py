import math
import os
import time
import argparse
from MixingNet import MixingNetwork
from QNet import QmixAgent

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from battle_env import MAgentBattle
from agent.agent_rl.agent_rl import AgentRL
from agent.agent_rule.agent_random import AgentRandom

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=5,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--dim-hidden", type=int, default=128,
                        help="hidden dim for QmixAgent")
    parser.add_argument("--mixing-embed-dim", type=int, default=64,
                        help="mixing_embed_dim for MixingNetwork")
    parser.add_argument("--target-network-update-freq", type=int, default=10, 
                        help="the frequency to update target network")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

args = parse_args()

logdir = 'logs'
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfile = f'{logdir}/lr{args.learning_rate}_q_hidden_dim{args.dim_hidden}_mixing_dim{args.mixing_embed_dim}_more_attack_log_st_0.01_atp_0.5_ed5m.txt'

env = MAgentBattle(visualize=False, eval_mode=False, obs_flat=True)
qmix_agents = [QmixAgent(env.dim_obs, args.dim_hidden, env.dim_action) for _ in range(env.num_agent)]
random_agents = AgentRandom(env.num_agent, env.dim_obs, env.dim_action)

num_rl_win = 0
num_random_win = 0
num_game = 0

(obs1, obs2), done, (valid1, valid2) = env.reset()

obs = torch.zeros((args.num_steps, env.num_agent, env.dim_obs))  # 각 단계 및 환경 인스턴스에서의 에이전트 관측값
actions = torch.zeros((args.num_steps, env.num_agent, env.dim_action))  # 각 단계 및 환경 인스턴스에서 취해진 에이전트의 행동
rewards = torch.zeros((args.num_steps, env.num_agent))  # 각 행동 후 환경으로부터 받은 보상
dones = torch.zeros((args.num_steps, env.num_agent))  # 각 단계 및 환경 인스턴스가 종료 상태로 이어지는지 여부
valids = torch.zeros((args.num_steps, env.num_agent))

global_step = 0  # 전체 환경에서 처리된 총 스텝 수
start_time = time.time()  # 훈련 과정 시작 시간
next_obs = torch.Tensor(obs1[0])  # 다음 관측값
next_done = torch.zeros(env.num_agent)  # 각 환경에서 다음 상태가 종료 상태인지 여부
next_valid = torch.ones(env.num_agent)
num_updates = args.total_timesteps // args.batch_size

epsilon_start = 1.0
epsilon_final = 0.001
epsilon_decay = 5000000  # Adjust this to control how fast epsilon decays
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

gamma = args.gamma  # 할인율
target_network_update_freq = args.target_network_update_freq  # 목표 네트워크 업데이트 빈도

# Target Mixing Network 초기화
dim_total_obs = env.num_agent * env.dim_obs
mixing_network = MixingNetwork(env.num_agent, dim_total_obs, args.mixing_embed_dim)
target_mixing_network = copy.deepcopy(mixing_network)

# Optimizer for the parameters of both agent networks and the mixing network
parameters = [param for agent in qmix_agents for param in agent.parameters()] + list(mixing_network.parameters())
optimizer = optim.Adam(parameters, lr=args.learning_rate, eps=1e-5)
winning_rates = []
# Main training loop
for update in range(num_updates):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow
    
    hidden_states = torch.zeros(env.num_agent, args.dim_hidden)  # GRU hidden states, initialize as needed
    next_hidden_states = []
    
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        valids[step] = next_valid
        epsilon = epsilon_by_frame(global_step)
        
        # Compute Q values for each agent
        q_values = []
        for agent_idx, agent in enumerate(qmix_agents):
            q_value, next_hidden_state = agent(obs[step][agent_idx], hidden_states[agent_idx])
            q_values.append(q_value)
            next_hidden_states.append(next_hidden_state)
            
        # Select actions for each agent using epsilon-greedy policy
        actions = []
        for agent_idx, q_val in enumerate(q_values):
            if np.random.uniform(0, 1) < epsilon:
                action = torch.randint(0, env.dim_action, (1,))
            else:
                action = q_val.argmax(dim=-1)
            actions.append(action)
        
        action1 = torch.Tensor(actions)
        
        # TODO : action2 random
        action2 = torch.Tensor(random_agents.get_action(obs2))

        # 환경을 한 스텝 진행
        (obs1, obs2), (reward1, reward2), (done1, done2, done_env), (valid1, valid2) = \
            env.step(action1.cpu().numpy().astype(np.int32), action2.cpu().numpy().astype(np.int32)) 
        
        if done_env:
            num_game += 1

            if len(obs1[1]) > len(obs2[1]):
                winstr = 'RL Win'
                num_rl_win += 1
            elif len(obs1[1]) < len(obs2[1]):
                winstr = 'Random Win'
                num_random_win += 1
            else:
                winstr = 'Draw'
            log_sentence = f'Update: {update}|, RL win: {num_rl_win / num_game:.3f} | Random win: {num_random_win / num_game:.3f}'
            print(log_sentence)
            winning_rates.append(num_rl_win / num_game)
            with open(logfile, 'a') as file:
                file.write(f'{log_sentence}\n')
            # print(action1)
            (obs1, obs2), (_, _, _), (_, _) = env.reset()
        
        rewards[step][torch.tensor(valid1, dtype=torch.bool)] = torch.tensor(reward1).view(-1)

        # 현재 및 다음 상태의 에이전트 Q값 준비
        current_q_val = []
        for idx, action in enumerate(actions):
            current_q_val.append(q_values[idx][action])
        current_agent_qs = torch.Tensor(current_q_val).unsqueeze(0)  # 현재 에이전트 Q값 준비
        current_total_state = torch.Tensor(obs1[0].flatten())  # 현재 전체 상태 결합
        
        # Calculate current Q values
        current_qs = mixing_network(current_agent_qs, current_total_state)
        
        next_obs, next_done, next_valid = torch.Tensor(obs1[0]), torch.Tensor(done1), torch.Tensor(valid1)
        
        # Compute Q values for each agent
        next_q_values = []
        for agent_idx, agent in enumerate(qmix_agents):
            next_q_value, _ = agent(obs[step][agent_idx], next_hidden_states[agent_idx])
            max_q_value = next_q_value.max(0)[0]
            next_q_values.append(max_q_value)
        next_agent_qs = torch.Tensor(next_q_values).unsqueeze(0)  # 다음 에이전트 Q값 준비
        next_total_state = torch.Tensor(next_obs.flatten())  # 다음 전체 상태 결합

        with torch.no_grad():
            target_qs = target_mixing_network(next_agent_qs, next_total_state)
            y_tot = rewards[step] + gamma * target_qs * (1 - dones)
        # Calculate loss
        loss = ((current_qs - y_tot) ** 2).mean()

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        hidden_state = torch.Tensor(next_hidden_state)

    # Update the target networks
    if update % target_network_update_freq == 0:
        target_mixing_network.load_state_dict(mixing_network.state_dict())
        

# Save the model
for agent_idx, agent in enumerate(qmix_agents):
    torch.save(agent.state_dict(), f'{logdir}/lr{args.learning_rate}_q_hidden_dim{args.dim_hidden}_mixing_dim{args.mixing_embed_dim}_agent_{agent_idx}_more_attack_st_0.01_atp_0.5_ed5m.pt')
torch.save(mixing_network.state_dict(), f'{logdir}/lr{args.learning_rate}_q_hidden_dim{args.dim_hidden}_mixing_dim{args.mixing_embed_dim}_mixing_network_more_attack_st_0.01_atp_0.5_ed5m.pt')
# save winning rates
np.save(f'{logdir}/lr{args.learning_rate}_q_hidden_dim{args.dim_hidden}_mixing_dim{args.mixing_embed_dim}_winning_rates_more_attack_st_0.01_atp_0.5_ed5m.npy', winning_rates)
