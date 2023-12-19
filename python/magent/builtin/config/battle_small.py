""" battle of two armies """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -0.005,  # 수동적 행동에 대한 패널티 증가
         'kill_reward': 20,     # 적 처치 보상 증가
         'dead_penalty': -0.1, 'attack_penalty': -0.5,
         })

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=4.0)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=4.0)
    
    return cfg
