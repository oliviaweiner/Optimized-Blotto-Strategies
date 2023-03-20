from blotto_game import BlottoAgent
import random
import networkx as nx
import pandas as pd
import numpy as np
import math

def create_bayesian_agent(num_towers, num_soldiers, bucketize):
    last_strategies = None
    move_history = []
    last_graph = None
    bucket_ranges_dict = calculate_bucket_ranges_dict(num_towers, num_soldiers) if bucketize else None
    parameters = [last_strategies, move_history, num_towers, num_soldiers, bucketize, last_graph, bucket_ranges_dict]
    return BlottoAgent(parameters, opening_strategy, informed_strategy)
    
def calculate_bucket_ranges_dict(num_towers, num_soldiers):
    output = {}
    for i in range(5):
        for j in range(num_towers):
            start = None
            stop = None
            for k in range(num_soldiers + 1):
                if score_bucket(k, j + 1, num_towers, num_soldiers) == i:
                    start = k
                    break
            if start != None:
                for k in range(start + 1, num_soldiers + 1):
                    if score_bucket(k, j + 1, num_towers, num_soldiers) != i:
                        stop = k
                        break
                if stop == None:
                    stop = num_soldiers + 1
            output[(i,j)] = (start, stop)
    return output
            
    
def update_info(parameters, strategy_batch, self_index):
    num_towers = parameters[2]
    num_soldiers = parameters[3]
    bucketize = parameters[4]
    strategy_batch_buckets = [bucketize_strategy(strategy, num_towers, num_soldiers) for strategy in strategy_batch] if bucketize else strategy_batch
    if parameters[0] != None:
        parameters[1].append(parameters[0] + strategy_batch_buckets)
    parameters[0] = strategy_batch_buckets
    
def bucketize_strategy(strategy, num_towers, num_soldiers):
    return [score_bucket(strategy[i], i + 1, num_towers, num_soldiers, True) for i in range(num_towers)]

def score_bucket(soldiers_assigned, tower_points, num_towers, num_soldiers):
    expected_points = tower_points * num_soldiers * 2 / (num_towers * (num_towers - 1))
    if not bucketize:
        return soldiers_assigned
    elif soldiers_assigned > 1.5 * expected_points:
        return 4
    elif soldiers_assigned > .75 * expected_points:
        return 3
    elif soldiers_assigned > 3:
        return 2
    elif soldiers_assigned > 0:
        return 1
    else:
        return 0

def make_strategy_correct(strategy, num_soldiers):
    while sum(strategy) < num_soldiers:
        chosen_tower = random.randint(0, len(strategy) - 1)
        strategy[chosen_tower] += 1
    while sum(strategy) > num_soldiers:
        chosen_tower = random.randint(0, len(strategy) - 1)
        if strategy[chosen_tower] != 0:
            strategy[chosen_tower] -= 1
    return strategy
    
def uniform_strategy(num_towers, num_soldiers):
    return make_strategy_correct([num_soldiers // num_towers] * num_towers, num_soldiers)
    
def opening_strategy(agent):
    num_towers = agent.parameters[2]
    num_soldiers = agent.parameters[3]
    return uniform_strategy(num_towers, num_soldiers)

def informed_strategy(agent, prev_round_strategies, junk1, junk2, self_index):
    update_info(agent.parameters, prev_round_strategies, self_index)
    move_history = agent.parameters[1]
    num_towers = agent.parameters[2]
    num_soldiers = agent.parameters[3]
    bucket_ranges_dict = agent.parameters[6]
    bucketize = agent.parameters[4]
    num_players = len(prev_round_strategies)
    #last_graph = agent.parameters[5]
    if agent.parameters[5] == None:
        agent.parameters[5] = initialize_graph(num_players, num_towers)
        return uniform_strategy(num_towers, num_soldiers)
    else:
        update_graph(agent.parameters[5], move_history, num_towers, num_players)
        return get_optimal_strategy(agent.parameters[5], move_history, bucketize, bucket_ranges_dict, self_index, num_towers, num_soldiers, num_players)

def initialize_graph(num_players, num_towers):
    G = nx.DiGraph()
    G.add_nodes_from(range(2 * num_players * num_towers))
    return G

def update_graph(G, move_history, num_towers, num_players):
    raw_data = pd.DataFrame(data=[element for sublist in move_history for element in sublist])
    mega_counts_dict = {}
    score = calculate_bayesian_score(G, raw_data, mega_counts_dict)
    local_maximum = False
    while True:
        new_score = find_improvement(G, score, num_towers * num_players, raw_data, mega_counts_dict)
        print(new_score)
        if new_score != None:
            score = new_score
        else:
            break

def calculate_bayesian_score(graph, raw_data, mega_counts_dict):
    #assert is_acyclic(graph)
    output = 0
    n = len(raw_data.columns)
    num_options = list(raw_data.nunique())
    for i in range(n):
        if i not in mega_counts_dict:
            mega_counts_dict[i] = {}
        parents = list(graph.predecessors(i))
        parent_tuple = tuple(parents)
        if parent_tuple in mega_counts_dict[i]:
            output += mega_counts_dict[i][parent_tuple]
        else:
            diff = 0
            counts_dict = {}
            for junk, row in raw_data.iterrows():
                parent_value_tuple = tuple(row[parents])
                if parent_value_tuple not in counts_dict:
                    counts_dict[parent_value_tuple] = {}
                node_value = row[i]
                if node_value not in counts_dict[parent_value_tuple]:
                    counts_dict[parent_value_tuple][node_value] = 1
                else:
                    counts_dict[parent_value_tuple][node_value] += 1
            for j in counts_dict:
                total_count = 0
                for k in counts_dict[j]:
                    count = counts_dict[j][k]
                    total_count += count
                    diff += math.lgamma(1 + count)
                diff -= math.lgamma(num_options[i] + total_count)
                diff += math.lgamma(num_options[i])
            output += diff
            mega_counts_dict[i][parent_tuple] = diff
    return output

def find_improvement(graph, score, flags_per_round, raw_data, mega_counts_dict):
    for i in range(flags_per_round):
        for j in range(flags_per_round, 2 * flags_per_round):
            if graph.has_edge(i, j):
                graph.remove_edge(i, j)
            else:
                graph.add_edge(i, j)
            bayesian_score = calculate_bayesian_score(graph, raw_data, mega_counts_dict)
            if bayesian_score > score:
                return bayesian_score
            elif graph.has_edge(i, j):
                graph.remove_edge(i, j)
            else:
                graph.add_edge(i, j)
    return None

def get_optimal_strategy(G, move_history, bucketize, bucket_ranges_dict, self_index, num_towers, num_soldiers, num_players):
    last_strategies = move_history[-1][:num_players]
    my_last_strategy = move_history[-1][self_index]
    prob_array = get_prob_array(G, move_history, last_strategies, bucketize, bucket_ranges_dict, num_towers, num_soldiers, num_players, self_index)
    return max_strategy_from_probs(prob_array, num_towers, num_soldiers, my_last_strategy)
    
def max_strategy_from_probs(prob_array, num_towers, num_soldiers, my_last_strategy):
    solution = my_last_strategy[:]
    score = calculate_expected_score(solution, prob_array)
    change = True
    while change:
        change = False
        for i in range(num_towers):
            if solution[i] != 0:
                for j in range(num_towers):
                    if i != j:
                        solution[i] -= 1
                        solution[j] += 1
                        new_score = calculate_expected_score(solution, prob_array)
                        if new_score > score:
                            change = True
                            score = new_score
                            print(score)
                        else:
                            solution[i] += 1
                            solution[j] -= 1
    return solution
    
def calculate_expected_score(strategy, prob_array):
    output = 0
    for tower_minus, troops in enumerate(strategy):
        tower = tower_minus + 1
        output += tower * (np.sum(prob_array[tower_minus][:troops]) + (prob_array[tower_minus][troops] / 2))
    return output
        

def get_prob_array(G, move_history, last_strategies, bucketize, bucket_ranges_dict, num_towers, num_soldiers, num_players, self_index):
    output = np.zeros((num_towers, num_soldiers + 1))
    for i in range(num_players):
        if i != self_index:
            for j in range(num_towers):
                next_allocation_index = (num_players * num_towers) + (i * num_towers) + j
                parent_indices = G.predecessors(next_allocation_index)
                counts_dict = {}
                total_counts = 0
                for ROW in move_history:
                    row = [element for sublist in ROW for element in sublist]
                    if check_row(row, next_allocation_index, parent_indices, last_strategies):
                        if row[next_allocation_index] not in counts_dict:
                            counts_dict[row[next_allocation_index]] = 1
                        else:
                            counts_dict[row[next_allocation_index]] += 1
                        total_counts += 1
                if total_counts > 0:
                    if not bucketize:
                        for val in counts_dict:
                            output[j][val] += counts_dict[val] / (total_counts * (num_players - 1))
                    else:
                        for val in counts_dict:
                            start, stop = bucket_ranges_dict[(val, j)]
                            for k in range(start, stop):
                                output[j][k] += counts_dict[val] / (total_counts * (num_players - 1) * (stop - start))
                else:
                    for k in range(num_soldiers + 1):
                        output[j][k] += 1 / ((num_soldiers + 1) * (num_players - 1))
                
    return output

def check_row(row, child_index, parent_indices, last_strategies):
    for p in parent_indices:
        if row[p] != row[last_strategies]:
            return False
    return True
