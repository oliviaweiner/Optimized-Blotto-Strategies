# Optimized-Blotto-Strategies

Colonel Blotto games are an enduring puzzle in game theory with many useful applications. Solving for a viable strategy can be a difficult task, and solving for the optimal strategy depends heavily on context and knowledge of one’s opponents. To attempt to compare various methods, we designed two kinds of agents (Deep Q-Learning agents and Bayesian Graphical Agents) utilizing different automated decision-making strategies and had them compete over 1000 iterations of blotto with our baseline agents. We found that over our initial 1000 iterations, the Bayesian Graphical Agents outperformed all other agents, whereas Deep Q-Learning Agents were only able to outperform deterministic baselines. By running a second experiment in which a Deep Q- Learning Agent was trained for 4000 iterations, we showed that the failure of the Deep Q-Learning Agent was likely a fluke of the limited data. We thus showed that under settings with fewer iterations, BGAs outperform Deep Q-Learning Agents.

Note: Our Deep Q-learning implementation used this tutorial: https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc as a baseline to improve upon.

Please refer to project report for more information.
