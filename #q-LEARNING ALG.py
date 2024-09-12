import numpy as np
import random

# Reward matrix
R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, -1, -1, -1, 0, 100]
])
print("Reward Matrix:")
print(R)

# Parameters
A = R.shape[1]  # Number of actions (columns in R)
S = R.shape[0]  # Number of states (rows in R)
Gd = 0.8  # Gamma (discount factor)
alpha = 0.1  # Learning rate
eps = 1000  # Number of episodes

# Initialize Q-table
Q = np.zeros((S, A))

# Q-learning algorithm
for episode in range(eps):
    state = random.randint(0, S-1)  # Start from a random state
    done = False
    while not done:
        # Choose an action (epsilon-greedy policy)
        if random.uniform(0, 1) < 0.1:  # Exploration
            possible_actions = np.where(R[state, :] != -1)[0]
            if len(possible_actions) > 0:
                action = random.choice(possible_actions)
            else:
                break
        else:  # Exploitation
            possible_actions = np.where(R[state, :] != -1)[0]
            if len(possible_actions) > 0:
                action = possible_actions[np.argmax(Q[state, possible_actions])]
            else:
                break

        # Take the action and observe the reward and next state
        next_state = np.argmax(R[state, :] != -1)
        reward = R[state, action]
        
        # Update Q-table using the Q-learning formula
        best_next_action = np.argmax(Q[next_state, :])
        Q[state, action] += alpha * (reward + Gd * Q[next_state, best_next_action] - Q[state, action])
        
        # Move to the next state
        state = next_state
        
        # Check if we reached a terminal state (no possible actions left)
        if len(np.where(R[state, :] != -1)[0]) == 0:
            done = True

print("Final Q-table:")
print(Q)



