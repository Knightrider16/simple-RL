import numpy as np

env = np.array([[0, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 2]])

actions = ['right', 'left', 'up', 'down']

i, j = 0, 0
state = env[i, j]
stop = False

def move(i, j, movement):
    if movement == 'right':
        if j == env.shape[1] - 1:
            print("Can't do that")
        else:
            j += 1
    elif movement == 'left':
        if j == 0:
            print("Can't do that")
        else:
            j -= 1
    elif movement == 'up':
        if i == 0:
            print("Can't do that")
        else:
            i -= 1
    elif movement == 'down':
        if i == env.shape[0] - 1:
            print("Can't do that")
        else:
            i += 1
    return i, j


while not stop:
    movement = np.random.choice(actions)
    i, j = move(i, j, movement)
    state = env[i, j]
    print(f"Agent moved {movement} to position ({i}, {j}), state = {state}")

    if state == 1:
        print("Agent has reached dead state. Resetting position to (0,0).")
        i, j = 0, 0
        state = env[i, j]
    elif state == 2:
        print("Agent has reached the goal. Stopping.")
        stop = True
