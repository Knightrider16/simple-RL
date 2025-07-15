import numpy as np

env = np.array([[0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 2]])

actions = ['right', 'left', 'up', 'down']
action_to_index = {a: i for i, a in enumerate(actions)}
alpha = 0.1 
gamma = 0.9 
epsilon = 0.2  
episodes = 1000

q_table = np.zeros((env.shape[0], env.shape[1], len(actions)))

def move(i, j, action):
    if action == 'right' and j < env.shape[1] - 1:
        j += 1
    elif action == 'left' and j > 0:
        j -= 1
    elif action == 'up' and i > 0:
        i -= 1
    elif action == 'down' and i < env.shape[0] - 1:
        i += 1
    return i, j

for episode in range(episodes):
    i, j = 0, 0
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(actions)
        else:
            action = actions[np.argmax(q_table[i, j])]

        new_i, new_j = move(i, j, action)
        reward = 0
        state = env[new_i, new_j]

        if state == 1:
            reward = -10
            done = True
        elif state == 2:
            reward = +10
            done = True
        else:
            reward = -1 

        old_q = q_table[i, j, action_to_index[action]]
        max_future_q = np.max(q_table[new_i, new_j])
        new_q = old_q + alpha * (reward + gamma * max_future_q - old_q)
        q_table[i, j, action_to_index[action]] = new_q

        i, j = new_i, new_j

i, j = 0, 0
print("\nTesting learned policy:")
while True:
    action = actions[np.argmax(q_table[i, j])]
    i, j = move(i, j, action)
    print(f"Moved {action} to ({i}, {j})")
    if env[i, j] == 2:
        print("Reached goal state")
        break
    elif env[i, j] == 1:
        print("Reached dead state")
        break

policy_table = np.full((4, 4),'', dtype = object)

for i in range(env.shape[0]):
    for j in range(env.shape[1]):
        if env[i, j] == 1:
            policy_table[i, j] = 'X' 
        elif env[i, j] == 2:
            policy_table[i, j] = 'G'
        else:
            best_action_index = np.argmax(q_table[i, j])
            policy_table[i, j] = actions[best_action_index]

print("\nPolicy Table:")
for row in policy_table:
    print('\t'.join(row))
