# Feature acquisition
Feature acquisition algorithms address the problem of acquiring informative features while balancing the costs of acquisition to improve the learning performances of ML models. Previous approaches have focused on calculating the expected utility values of features to determine the acquisition sequences. Other approaches formulated the problem as a Markov Decision Process (MDP) and applied reinforcement learning based algorithms. In comparison to previous approaches, we focus on 1) formulating the feature acquisition problem as a MDP and applying Monte Carlo Tree Search, 2) calculating the intermediary rewards for each acquisition step based on model improvements and acquisition costs and 3) simultaneously optimizing model improvement and acquisition costs with multi-objective Monte Carlo Tree Search. With Proximal Policy Optimization and Deep Q-Network algorithms as benchmark, we show the effectiveness of our proposed approach with experimental study. 

1. train_classifiers.py contains script to train a logistic regression or CNN classifier based on given random state and seed value. 
    ```diff
      python3 train_classifiers.py --help
    ```
2. ppo_mnist.py
    ```diff
      python3 train_classifiers.py --help
    ```
3. ddqn_mnist.py
    ```diff
      python3 train_classifiers.py --help
    ```
4. so_mcts_mnist.py
    ```diff
      python3 so_mcts_mnist.py
    ```
5. mcts_test_mnist_mo.py
    ```diff
      python3 mo_mcts_mnist.py
    ```
