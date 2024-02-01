

import numpy as np
import tensorflow as tf


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


# Define the neural network to be used by the agent in the environment.
# This model takes the position directly from the environment.
# It outputs 5 values, each representing an approximation of the Q-value of each action that can be taken in the environment.
# (There are five actions because move_direction can be either -2, -1, 0, 1, or 2.)
# The agent uses this output by taking the action with the largest approximated Q-value.
# Training this model increases the accuracy of the approximations over time, resulting in greater accumulation of reward.
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((2,9)), # Each position in the environment is an array of this shape.
        tf.keras.layers.Flatten(), # Flatten it to a 1D array of 18 elements.
        tf.keras.layers.Dense(32, 'relu'), # Dense layers...
        tf.keras.layers.Dense(32, 'relu'),
        tf.keras.layers.Dense(5), # The output layer with linear activation, producing an approximation of the Q-values of the five actions.
    ])
    model.compile('adam', 'msle')
    # Mean squared error is fairly standard, but mean squared logarithmic error can be better for certain reward functions such as the one defined above.
    # Adam optimzer is also pretty standard. RMSProp may be better for recurrent neural networks.
    return model



###############################################################
model = build_model()
env = SuperSimpleEnvironment()

old_states = []
new_states = []
actions = []
rewards = []

print(env.position)
while True:
    print('-'*57)
    action = np.argmax(model.predict(np.expand_dims(env.position, 0)))
    # Keras models assume there's a batch dimension, so the position is nested in another array so that it is at index 0 along the batch dimension.
    print(f'Move by {action-2:+}')
    old_state = env.position.copy()
    env.step(action - 2)
    # Subtract 2 so that the 0th action is -2, 1st action is -1, 2nd action is 0, 3rd action is 1, and 4th action is 2.
    new_state = env.position.copy()
    print(env.position)
    reward = get_reward(env.position)
    # Eventually, the reward received will consistently be 0.5 and 1.0, after sufficient training.
    print(f'Received reward {reward:.4f}')

    # Save the transition from old_state to new_state, which includes information about the reward received by taking some action.
    old_states.append(old_state)
    new_states.append(new_state)
    actions.append(action)
    rewards.append(reward)

    # Perform experience replay.
    # This is where the model trains on a random batch of transitions that have been saved.

    # First, gather the random batch of transitions.
    transition_indices = np.random.randint(0, len(old_states), size=16)
    sample_old_states = np.array(old_states)[transition_indices]
    sample_new_states = np.array(new_states)[transition_indices]
    sample_actions = np.array(actions)[transition_indices]
    sample_rewards = np.array(rewards)[transition_indices]

    # Next, use the update rule to train the model on its own predictions, but with one of them slightly more accurate than before.
    pred_old_states = model.predict(sample_old_states)
    pred_new_states = model.predict(sample_new_states)
    pred_old_states[np.arange(16),sample_actions] = sample_rewards + 0.98 * np.max(pred_new_states, axis=1)

    model.fit(sample_old_states, pred_old_states, verbose=False)



###############################################################
# model = build_model()
# env = SuperSimpleEnvironment()

# print(env.position)
# for i in range(20):
#     print('-'*57)
#     action = np.argmax(model.predict(np.expand_dims(env.position, 0)))
#     # Keras models assume there's a batch dimension, so the position is nested in another array so that it is at index 0 along the batch dimension.
#     print(f'Move by {action:+}')
#     env.step(action - 2)
#     # Subtract 2 so that the 0th action is -2, 1st action is -1, 2nd action is 0, 3rd action is 1, and 4th action is 2.
#     print(env.position)
#     reward = get_reward(env.position)
#     print(f'Received reward {reward:.4f}')



###############################################################
# env = SuperSimpleEnvironment()
# print(env.position)
# for i in range(20):
#     print('-'*57)
#     action = np.random.randint(-2,3)
#     print(f'Move by {action:+}')
#     env.step(action)
#     print(env.position)
#     reward = get_reward(env.position)
#     print(f'Received reward {reward:.4f}')



###############################################################
# env = SuperSimpleEnvironment()
# print(env.position)
# for i in range(20):
#     print('-'*57)
#     action = np.random.randint(-2,3)
#     print(f'Move by {action:+}')
#     env.step(action)
#     print(env.position)


