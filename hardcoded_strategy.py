from blotto_game import BlottoAgent
import random

def create_hardcoded_agent(strategy_list, num_towers, num_soldiers, weights=None):
    return BlottoAgent([strategy_list, weights], hardcoded_first_strategy, hardcoded_strategy)
    
def hardcoded_strategy(agent, junk1, junk2, junk3, junk4):
    return hardcoded_first_strategy(agent)
    
def hardcoded_first_strategy(agent):
    return random.choices(agent.parameters[0], weights=agent.parameters[1])[0]
