
import gymnasium as gym
import numpy as np
import tensorflow as tf



env = gym.make("CartPole-v1", render_mode='human')

print(f'State space shape: {env.observation_space.shape}')
print(f'Action space size: {env.action_space.n}')



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(env.observation_space.shape),
        tf.keras.layers.Dense(64, 'relu'),
        # tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(64, 'relu'),
        # tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(64, 'relu'),
        # tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(env.action_space.n),
    ])
    model.compile('adam', 'mse')
    return model



model = build_model()
model.optimizer.learning_rate = 0.00001
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())


epsilon_min = 0.01
epsilon_decay = 0.99
epsilon_delay = 128 # amount of steps are simulated before epsilon drops from 1.0
tau = 0.75 # target model update speed
gamma = 0.99 # preference for long-term reward
memory_capacity = 1000000

state, info = env.reset()
old_states = np.empty((memory_capacity, *state.shape))
new_states = np.empty(old_states.shape)
actions = np.empty(memory_capacity, dtype=int)
rewards = np.empty(memory_capacity)
terminals = np.empty(memory_capacity, dtype=bool)
total_steps = 0
epsilon = 1
do_training = False
episode_reward = 0
while True:
    if do_training and total_steps >= epsilon_delay:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if np.random.random() < epsilon:
        action = np.random.randint(env.action_space.n)
    else:
        action = np.argmax(model(np.expand_dims(state, 0)).numpy())

    new_state, reward, terminal, truncated, info = env.step(action)

    if reward != 0:
        do_training = True
        episode_reward += reward

    if truncated:
        terminal = True # treat truncation equally as termination in DQL

    print(f'Epsilon: {epsilon:.2f} | Reward: {reward:+.2f} | Action: {action} | Episode Reward = {episode_reward:+.2f}')

    if terminal:
        state, info = env.reset()
        episode_reward = 0

    i = total_steps % memory_capacity
    old_states[i] = state
    new_states[i] = new_state
    actions[i] = action
    rewards[i] = reward
    terminals[i] = terminal

    if do_training:
        mem_index = min(total_steps + 1, memory_capacity)

        # an experiment of weighting outlier rewards more heavily (similar to PER, Prioritized Experienced Replay)
        # rs = rewards[:mem_index]
        # r_mu = np.mean(rs)
        # r_sigma = np.std(rs)
        # if r_sigma == 0:
        #     r_sigma = 1
        # r_sigma_inv = 1. / r_sigma

        transition_indices = np.random.randint(0, mem_index, size=256)
        sample_old_states = old_states[transition_indices]
        sample_new_states = new_states[transition_indices]
        sample_actions = actions[transition_indices]
        sample_rewards = rewards[transition_indices]
        sample_terminals = terminals[transition_indices]

        pred_old_states = model(sample_old_states).numpy()
        pred_new_states = model(sample_new_states).numpy()
        target_pred_new_states = target_model(sample_new_states).numpy()
        pred_old_states[np.arange(len(transition_indices)),sample_actions] = sample_rewards + gamma * target_pred_new_states[np.arange(len(transition_indices)), np.argmax(pred_new_states, axis=1)] * (1 - sample_terminals.astype(int))

        model.fit(sample_old_states, pred_old_states, verbose=False,
            # PER-like experiment
            # sample_weight=np.array([
            #     # compute z-score for each reward
            #     (r - r_mu) * r_sigma_inv
            #     for r in sample_rewards
            # ]),
        )



        # two ways to update the target model: interpolation or periodic updates
        # both could theoretically be used simulataneously, but it probably wouldn't have any meaningful results

        # interpolation method
        # target_model.set_weights([w0 * tau + w1 * (1 - tau) for w0, w1 in zip(model.get_weights(), target_model.get_weights())])

        # periodic update method
        if total_steps % 128 == 0:
            target_model.set_weights(model.get_weights())

    total_steps += 1
    state = new_state


