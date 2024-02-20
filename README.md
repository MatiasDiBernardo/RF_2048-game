## AI plays 2048

Reinforcement Learning project where an AI-based agent learns to play the game 2048. The game gui is developed in pygame and the agent is implemented with PyTorch.  

# Model and Reward

The model learns based on Bellmans Equation and Neural Q Value. I tried different reward functions, but the best one gives rewards when the maximum block value increases and penalization when there are no possible moves. The best results for the linear layers where with two  hidden layers of sizes 1024 and 256. The input is a binaty input of the board (vector of size 256) and the output is the movement represented by a state binary vector (size 4). All the layers use Relu activation functions. Architecture resume: (256 -> 1024 -> 256 -> 4)

# Results 

The best result with this Neural approach was a max score of 1430 points. Trained for 4000 games and the mean score was 473 points. The agent is not able to consistently reach the 1024 block because it is not able to create a general strategy that accounts for the randomness of the game. I try to guide the agent with more precise penalization and rewards, but the results are the same. The main problem is that by the nature of the game, the agent can develop a strategy valid for the early state of the game that no longer holds in the long run.