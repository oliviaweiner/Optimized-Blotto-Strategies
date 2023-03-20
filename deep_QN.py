import tensorflow as tf
import numpy as np
from tensorflow import keras

import blotto_game
from collections import deque
import time
import random

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def create_model(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def random_strategy(agent):
    strategy = [0 for i in range(agent.n_towers)]
    for i in range(agent.n_soldiers):
        tower = random.randint(0, agent.n_towers - 1)
        strategy[tower] += 1
    return strategy

def partitions(n, k):
    output = []
    for c in itertools.combinations(range(n+k-1), k-1):
        output.append([b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))])
    return output

def action_dicts(n_soldiers, n_towers):
    n_actions = math.comb(n_soldiers + n_towers - 1, n_soldiers)
    act_to_int = {}
    int_to_act = {}
    for i, arr in enumerate(partitions(n_soldiers, n_towers)):
        act_to_int[str(arr)] = i
        int_to_act[i] = arr
    return (n_actions, act_to_int, int_to_act)


class DeepQNagent:

    def __init__(self, n_towers, n_soldiers, create_model_fn):
        self.epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
        self.max_epsilon = 1 # You can't explore more than 100% of the time
        self.min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
        self.decay = 0.01

        (self.n_actions, self.act_to_int_d, self.int_to_act_d) = action_dicts(n_soldiers, n_towers)

        self.model = create_model_fn((n_towers,), n_towers)
        self.target_model = create_model_fn((n_towers,), n_towers)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=50_000)
        self.steps_to_update_target_model = 0
        self.total_steps = 0
        self.total_training_rewards = 0

        self.n_towers = n_towers
        self.n_soldiers = n_soldiers

        self.last_observation = None
        self.last_action = None
        self.last_reward = None


    def get_qs(self, state, step):
        return self.model.predict(state.reshape([1, state.shape[0]]))[0]

    def train(self):
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618

        MIN_REPLAY_SIZE = 100
        if len(self.replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64
        mini_batch = random.sample(self.replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_states2 = current_states.sum(axis=1)
        current_qs_list = self.model.predict(current_states2)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        new_current_states2 = new_current_states.sum(axis=1)
        future_qs_list = self.target_model.predict(new_current_states2)

        X = []
        Y = []
        #change below iterations
        for index, (observation, action, reward, new_observation) in enumerate(mini_batch):
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            observation2 = np.array(observation)
            observation3 = observation2.sum(axis=0)
            X.append(observation3)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def create_DQN_agent(strategy_list, num_towers, num_soldiers, weights=None):
    DQN_agent = DeepQNagent(num_towers, num_soldiers, create_model)
    return blotto_game.BlottoAgent(DQN_agent, DQN_first_strategy, DQN_strategy)


def DQN_first_strategy(self_agent):
    DQN_agent = self_agent.parameters
    return random_strategy(DQN_agent)

def DQN_strategy(self_agent, prev_round_strategies, prev_round_scores, prev_round_wins, self_index):
    DQN_agent = self_agent.parameters
    observation = prev_round_strategies
    if DQN_agent.last_observation != None:
        DQN_agent.replay_memory.append([DQN_agent.last_observation, DQN_agent.last_action, DQN_agent.last_reward, observation])
    DQN_agent.last_observation = observation
    DQN_agent.last_action = observation[self_index]
    DQN_agent.last_reward = prev_round_scores[self_index]

    DQN_agent.steps_to_update_target_model += 1
    DQN_agent.total_steps += 1
    random_number = np.random.rand()

    # 1. Update the Main Network using the Bellman Equation
    if DQN_agent.steps_to_update_target_model % 4 == 0:
        DQN_agent.train() #check if correct format

    # 2. Explore using the Epsilon Greedy Exploration Strategy
    if random_number <= DQN_agent.epsilon:
        # Explore
        action = random_strategy(DQN_agent)

    else:
        # Exploit best known action
        # model dims are (batch, env.observation_space.n)
        encoded = np.array(observation)
        encoded2 = encoded.sum(axis=0)
        encoded_reshaped = encoded2.reshape([1, encoded2.shape[0]])
        predicted = DQN_agent.model.predict(encoded_reshaped).flatten()
        action_num = np.argmax(predicted)
        action = DQN_agent.int_to_act_d[action_num]

    if DQN_agent.steps_to_update_target_model >= 100:
        print('Copying main network weights to the target network weights')
        DQN_agent.target_model.set_weights(DQN_agent.model.get_weights())
        DQN_agent.steps_to_update_target_model = 0
        DQN_agent.epsilon = DQN_agent.min_epsilon + (DQN_agent.max_epsilon - DQN_agent.min_epsilon) * np.exp(-DQN_agent.decay * DQN_agent.total_steps)

    return action