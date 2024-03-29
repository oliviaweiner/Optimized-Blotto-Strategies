import numpy as np
import threading

class BlottoAgent:
    
    def __init__(self, initial_parameters, opening_strategy_func, strategy_func):

        self.parameters = initial_parameters
        
        self.opening_strategy_func = opening_strategy_func
        self.strategy_func = strategy_func
        
    def get_opening_strategy(self):
        return self.opening_strategy_func(self)
    
    def get_next_strategy(self, prev_round_strategies, prev_round_scores, prev_round_wins, self_index):
        return self.strategy_func(self, prev_round_strategies, prev_round_scores, prev_round_wins, self_index)
    
def simulate_blotto_game(agents, num_rounds, num_soldiers):
    strategies = [agent.get_opening_strategy() for agent in agents]
    historical_scores = []
    historical_wins = []
    for r in range(num_rounds):
        if r % 50 == 0:
            print('ROUND ' + str(r))
        for strat in strategies:
            assert sum(strat) == num_soldiers
        scores, wins = calculate_scores_and_wins(strategies)
        historical_scores.append(scores)
        historical_wins.append(wins)
        next_strategies = [None] * len(agents)
        threadies = []
        for i, agent in enumerate(agents):
            thready = threading.Thread(target=fill_with_strategy, args=(next_strategies, agent.get_next_strategy, strategies, scores, wins, i))
            threadies.append(thready)
            thready.start()
        for thready in threadies:
            thready.join()
        strategies = next_strategies
    listy = [historical_scores, historical_wins, average_scores_and_wins(historical_scores, historical_wins)]
    return listy
    
def fill_with_strategy(list, strategy_func, strategies, scores, wins, i):
    list[i] = strategy_func(strategies, scores, wins, i)

def calculate_scores_and_wins(strategies):
    num_players = len(strategies)
    num_towers = len(strategies[0])
    scores = [0] * num_players
    wins = [0] * num_players
    for i in range(num_players):
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
    average_scores = np.array(historical_scores).mean(axis=0)
    average_wins = np.array(historical_wins).mean(axis=0)
    return average_scores.tolist(), average_wins.tolist()
