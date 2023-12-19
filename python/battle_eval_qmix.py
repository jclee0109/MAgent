import time

import torch
import numpy as np

from MixingNet import MixingNetwork
from QNet import QmixAgent

from battle_env import MAgentBattle
from agent.agent_rl.agent_rl import AgentRL
from agent.agent_rule.agent_random import AgentRandom

def get_action_eval(model, obs, hidden_state, epsilon=0.001):
    # 모델을 평가 모드로 설정
    model.eval()
    with torch.no_grad():
        q_value, next_hidden_state = model(obs, hidden_state)
        # 최대 Q 값을 가진 행동 선택
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(env.dim_action)
        else:
            action = q_value.argmax(dim=-1).cpu().numpy()
    return action, next_hidden_state

if __name__ == "__main__":
    env = MAgentBattle(visualize=True, eval_mode=False, obs_flat=True)
    dim_total_obs = env.num_agent * env.dim_obs
    num_steps = 512
    num_envs = 2
    dim_hidden = 128
    dim_mixing_embed = 64
    # Load the trained models
    mixing_network = MixingNetwork(env.num_agent, dim_total_obs, dim_mixing_embed)
    mixing_network_state_dict = torch.load('logs/lr0.05_q_hidden_dim128_mixing_network.pt')
    mixing_network.load_state_dict(mixing_network_state_dict)
    agent_networks = [QmixAgent(env.dim_obs, dim_hidden, env.dim_action) for _ in range(env.num_agent)]
    agent2 = AgentRandom(env.num_agent, env.dim_obs, env.dim_action)

    for i in range(env.num_agent):
        agent_networks[i].load_state_dict(torch.load(f'logs/lr0.05_q_hidden_dim128_agent_{i}.pt'))
    
    # Set the networks to evaluation mode
    mixing_network.eval()

    (obs1, obs2), (done1, done2, done_env), (valid1, valid2) = env.reset()
    obs = torch.zeros((num_steps, env.num_agent, env.dim_obs))  # 각 단계 및 환경 인스턴스에서의 에이전트 관측값
    start = time.time()
    next_obs = torch.Tensor(obs1[0])  # 다음 관측값

    hidden_states = torch.zeros(env.num_agent, dim_hidden)
    next_hidden_states = []
    
    for step in range(0, num_steps):
        obs[step] = next_obs
        a1 = []
        for agent_idx, agent in enumerate(agent_networks):
            action, next_hidden_state = get_action_eval(agent, torch.Tensor(obs[step][agent_idx]), hidden_states[agent_idx])
            next_hidden_states.append(next_hidden_state)
            a1.append(action)
        a1 = np.array(a1, dtype=np.int32)
        # Team 2 make decisions. (in a decentralized manner)
        # Assuming agent2 uses random actions for evaluation
        a2 = agent2.get_action(obs2)

        (obs1, obs2), reward, (done1, done2, done_env), (valid1, valid2) = env.step(a1, a2)
        next_obs = torch.Tensor(obs1[0])
        # 종료 조건 체크
        if done_env:
            print(f"The winner is {'RL' if len(obs1[1]) > len(obs2[1]) else 'Random' if len(obs1[1]) < len(obs2[1]) else 'Draw'}")
            break
        hidden_state = torch.Tensor(next_hidden_state)
    env.close()
    print(f"Episode completed in {step} timesteps")
    
