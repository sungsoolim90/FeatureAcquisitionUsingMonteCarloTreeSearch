# Feature acquisition
Feature acquisition algorithms address the problem of acquiring informative features while balancing the costs of acquisition to improve the learning performances of ML models. Previous approaches have focused on calculating the expected utility values of features to determine the acquisition sequences. Other approaches formulated the problem as a Markov Decision Process (MDP) and applied reinforcement learning based algorithms. In comparison to previous approaches, we focus on 1) formulating the feature acquisition problem as a MDP and applying Monte Carlo Tree Search, 2) calculating the intermediary rewards for each acquisition step based on model improvements and acquisition costs and 3) simultaneously optimizing model improvement and acquisition costs with multi-objective Monte Carlo Tree Search. With Proximal Policy Optimization and Deep Q-Network algorithms as benchmark, we show the effectiveness of our proposed approach with experimental study. 

1. train_classifiers.py: Train a logistic regression or CNN classifier given random state and seed value. 
    ```diff
      python3 train_classifiers.py --help
      python3 train_classifiers.py --rs 1 --sv 12321 --mt cnn --st False --bs 256 -e 101
    ```
2. ppo_mnist.py: Train the PPO algorithm for feature acquition. 
    ```diff
      python3 ppo_mnist.py --help
      python3 ppo_mnist.py --rs 1 --sv 12321 --mt cnn --uf 49 --e 10 --st False
    ```
3. dqn_mnist.py: Train the DQN algorithm for feature acquition. 
    ```diff
      python3 dqn_mnist.py --help
      python3 dqn_mnist.py --rs 1 --sv 12321 --mt cnn --e 10 --st False
    ```
4. so_mcts_mnist.py: Train the single-objective MCTS algorithm for feature acquition. 
    ```diff
      python3 so_mcts_mnist.py
    ```
5. mo_mcts_mnist.py: Train the multi-objective MCTS algorithm for feature acquition. 
    ```diff
      python3 mo_mcts_mnist.py
    ```