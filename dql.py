

import numpy as np



# First, define an environment in which the agent will learn to maximize reward.

class SuperSimpleEnvironment:
    def __init__(self):
        self.reset()
    def reset(self):
        # Return the environment to the starting state.
        self.position = np.array([
            [False, False, False, False, False, True, False, False, False], # The "True" position is the target position.
            [True, False, False, False, False, False, False, False, False], # The "True" position is the current position.
        ], dtype=bool)
        self.total_steps = 0
        # The goal is to move the current position under the target position by moving left, moving right, or staying in place.
    def step(self, move_direction):
        # Every step, the target position either moves left, moves right, or stays in place.
        # It cannot move left if it's all the way on the left,
        # and it cannot move right if it's all the way on the right.

        target_direction = np.random.randint(-1,2) # -1, 0, or 1
        if target_direction == 1 and self.position[0,-1] or target_direction == -1 and self.position[0,0]:
            target_direction = 0
        self.position[0] = np.roll(self.position[0], target_direction) # Move the target position.

        # Supply the move_direction argument to move the current position.
        # move_direction is allowed to be -2, -1, 0, 1, or 2.
        cur_pos = np.argmax(self.position[1])
        if cur_pos + move_direction >= 9:
            move_direction = 8 - cur_pos
        elif cur_pos + move_direction < 0:
            move_direction = -cur_pos
        self.position[1] = np.roll(self.position[1], move_direction) # Move the current position.


# Now, define the reward function.
# This one is defined as the inverse of the distance (offset by 1) between the current position and the target position.
def get_reward(position):
    return 1. / (abs(np.argmax(position[0]) - np.argmax(position[1])) + 1)



env = SuperSimpleEnvironment()

print(env.position)
for i in range(20):
    print('-'*57)
    action = np.random.randint(-2,3)
    print(f'Move by {action:+}')
    env.step(action)
    print(env.position)
    reward = get_reward(env.position)
    print(f'Received reward {reward:.4f}')



###############################################################
# env = SuperSimpleEnvironment()
# print(env.position)
# for i in range(20):
#     print('-'*57)
#     action = np.random.randint(-2,3)
#     print(f'Move by {action:+}')
#     env.step(action)
#     print(env.position)


