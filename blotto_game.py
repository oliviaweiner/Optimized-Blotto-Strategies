import numpy as np

class BlottoAgent:
    
    def __init__(self, num_players, num_towers, num_soldiers, setup_func, opening_strategy_func, strategy_func):
        #initialize empty list to be filled with state parameters
        self.parameters = []
        self.num_towers = num_towers
        self.num_players = num_players
        self.num_soldiers = num_soldiers
        #fill parameters with necessary state variables according to game constraints
        setup_func(self.parameters, num_players, num_towers, num_soldiers)
        self.opening_strategy_func = opening_strategy_func
        self.strategy_func = strategy_func
        
    def get_opening_strategy(self):
        return opening_strategy_func(self)
    
    def get_next_strategy(prev_round_strategies, prev_round_scores, prev_round_wins, self_index):
        return strategy_func(self, prev_round_strategies, prev_round_scores, prev_round_wins, self_index)
    
def simulate_blotto_game(agents, num_towers, num_soldiers, num_rounds):
    strategies = [agent.get_opening_strategy() for agent in agents]
    cumulative_scores = [0] * num_players
    cumulative_wins = [0] * num_players
    historical_scores = []
    historical_wins = []
    for r in range(num_rounds):
        scores, wins = calculate_scores_and_wins(strategies)
        historical_scores.append(scores)
        historical_wins.append(wins)
        strategies = [agent.get_next_strategy(strategies, scores, wins, i) for i, agent in enumerate(agents)]
    return historical_scores, historical_wins
    
def calculate_scores_and_wins(strategies):
    num_players = len(strategies)
    num_towers = len(strategies[0])
    scores = [0] * num_players
    wins = [0] * num_players
    for i in range(num_players)
        for j in range(i + 1, num_players):
            iscore = 0
            jscore = 0
            for k in range(num_towers):
                if strategies[i][k] > strategies[j][k]:
                    iscore += k + 1
                elif strategies[i][k] < strategies[j][k]:
                    jscore += k + 1
                else:
                    iscore += (k+1)/2
                    jscore += (k+1)/2
            scores[i] += iscore
            scores[j] += jscore
            if iscore > jscore:
                wins[i] += 1
            elif iscore < jscore:
                wins[j] += 1
            else:
                wins[i] += .5
                wins[j] += .5
    return scores, wins
            
def average_scores_and_wins(historical_scores, historical_wins):
    pass
