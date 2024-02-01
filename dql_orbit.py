
'''
This script is based on dql.py, but it runs a different environment.

This environment can be rendered, and a few changes were made to either simplify the code or enable the extension of the DQL algorithm to the environment (e.g. implement terminal state handling).

Also, the epsilon-greedy policy is used instead of the greedy policy because it is a critical step in making the agent actually learn anything.
'''


import numpy as np
import tensorflow as tf
import cv2 # For rendering the environment

from dql_orbit_util import Object2D, simulate_gravity # ChatGPT-written object and function



class OrbitEnvironment:
    '''
    Simulate the control of an orbit around an astronomical object and reaching a target orbit.
    '''
    def __init__(self):
        self.death_radius = 0.5 # The agent cannot get closer to the object than this.  If it does, it dies and resets the environment.
        self.escape_radius = 3 # The agent cannot get farther from the object than this.  If it does, it dies and resets the environment.
        self.time_scale = 0.05 # Also known as "dt".
        self.gravitational_constant = 1e-4
        self.initial_satellite_velocity = 0.3165
        self.obj_planet = Object2D(mass=1000, x=0, y=0, vx=0, vy=0)
        self.obj_satellite = Object2D(mass=1, x=1, y=0, vx=0, vy=self.initial_satellite_velocity) # Unrealistic ratio of mass, given that the agent controls thrust and whatnot, but it makes simulation more interesting.
        self.stop = False # Modified within OrbitEnvironment.render() to signal when to stop the training loop
        self.reset()
    def thrust(self, x: float, y: float):
        # Thrusting takes fuel, and fuel is a limited resource.  If the agent runs out of fuel, it dies eventually because it will almost certainly not be in perfect orbit.
        # A higher reward is obtained by optimizing thrust towards orbit.
        if self.fuel > 0:
            self.obj_satellite.vx += x * self.time_scale
            self.obj_satellite.vy += y * self.time_scale
            self.fuel -= 1
    def reset(self):
        self.obj_planet.x = 0
        self.obj_planet.y = 0
        self.obj_planet.vx = 0
        self.obj_planet.vy = 0
        self.obj_satellite.x = 1
        self.obj_satellite.y = 0
        self.obj_satellite.vx = 0
        self.obj_satellite.vy = self.initial_satellite_velocity
        self.target_radius = 1
        self.current_radius = 1
        self.fuel = 50 # A hard limit, the agent has this many attempts to thrust into perfect orbit.
        self.terminal = False
    def step(self, action: int):
        if action == 0:
            # no thrust
            pass
        elif action == 1:
            # thrust in positive X
            self.thrust(1, 0)
        elif action == 2:
            # ...
            self.thrust(0, 1)
        elif action == 3:
            self.thrust(-1, 0)
        else:
            self.thrust(0, -1)
        simulate_gravity(self.obj_planet, self.obj_satellite, 3, self.time_scale, G=self.gravitational_constant)
        self.current_radius = ((self.obj_planet.x - self.obj_satellite.x) * (self.obj_planet.x - self.obj_satellite.x) + (self.obj_planet.y - self.obj_satellite.y) * (self.obj_planet.y - self.obj_satellite.y)) ** .5
        if self.current_radius <= self.death_radius or self.current_radius >= self.escape_radius:
            # The agent died!
            self.terminal = True
    def get_reward(self) -> float:
        return 1 / (1 + abs(self.current_radius - self.target_radius)) - 0.5
    def get_state(self) -> np.ndarray:
        return np.array([
            self.obj_satellite.x, self.obj_satellite.y, self.obj_satellite.vx, self.obj_satellite.vy,
            self.obj_planet.x, self.obj_planet.y, self.obj_planet.vx, self.obj_planet.vy,
            self.fuel * .1 - 2.5,
            int(self.fuel > 0) * 2 - 1, # A flag for telling the model whether it truly is out of fuel, since the interpolated values of nearly empty fuel and actually empty fuel are hard to distinguish.
        ])
    def render(self):
        img = np.zeros((128, 128))
        space_scale = 20
        cv2.circle(img, (int(64.5+space_scale*self.obj_planet.x), int(64.5-space_scale*self.obj_planet.y)), int(space_scale*self.obj_planet.mass**.5/64), (1, 1, 1), -1)
        cv2.circle(img, (int(64.5+space_scale*self.obj_satellite.x), int(64.5-space_scale*self.obj_satellite.y)), int(space_scale*self.obj_satellite.mass**.5/64), (1, 0, 0), -1)
        cv2.imshow('env', img)
        inp = cv2.waitKey(1000//60)
        if inp & 0xFF == 27: # escape key
            self.stop = True




def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((10,)),
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(24),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(16),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(5),
    ])
    model.compile('adam', 'mse')
    return model



model = build_model()
env = OrbitEnvironment()

old_states = []
new_states = []
actions = []
rewards = []
terminals = []

epsilon_period = 1024
force_reset_period = 2048 # sometimes an elliptical orbit is achieved and the simulation runs forever

total_steps = 0
while not env.stop:
    env.render() # Can be commented out to improve simulation speed, but performance cannot be visualized unless the model is saved and loaded into the environment with rendering enabled.

    old_state = env.get_state()

    epsilon = ((epsilon_period - total_steps) % epsilon_period / epsilon_period) ** 2
    if env.fuel == 0 or np.random.random() < epsilon:
        # no thrusting actions are effective with zero fuel, but we still want the model to observe that those actions are ineffective so that its learning is more stable
        action = np.random.randint(5)
    else:
        action = np.argmax(model(np.expand_dims(old_state, 0)).numpy())

    env.step(action)

    new_state = env.get_state()
    reward = env.get_reward()
    terminal = env.terminal

    if (total_steps + 1) % force_reset_period == 0:
        terminal = True

    fuel_str = '\u25a0' * env.fuel + '\u25a1' * (50 - env.fuel)
    print(f'Fuel: {env.fuel:02d} / 50 : {fuel_str} | Radius: {env.current_radius:.2f} || Epsilon: {epsilon:.2f} | Reward: {reward:+.2f} | Action: {action}')

    if terminal:
        env.reset()

    old_states.append(old_state)
    new_states.append(new_state)
    actions.append(action)
    rewards.append(reward)
    terminals.append(terminal)

    transition_indices = np.random.randint(0, len(old_states), size=16)
    sample_old_states = np.array(old_states)[transition_indices]
    sample_new_states = np.array(new_states)[transition_indices]
    sample_actions = np.array(actions)[transition_indices]
    sample_rewards = np.array(rewards)[transition_indices]
    sample_terminals = np.array(terminals)[transition_indices]

    pred_old_states = model(sample_old_states).numpy()
    pred_new_states = model(sample_new_states).numpy()
    pred_old_states[np.arange(len(transition_indices)),sample_actions] = sample_rewards + 0.98 * np.max(pred_new_states, axis=1) * (1 - sample_terminals.astype(int))

    model.fit(sample_old_states, pred_old_states, verbose=False)

    total_steps += 1


