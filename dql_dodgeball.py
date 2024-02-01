
'''
This script is based on dql.py, but it runs a different environment.

This environment can be rendered, and a few changes were made to either simplify the code or enable the extension of the DQL algorithm to the environment (e.g. implement terminal state handling).

Also, the epsilon-greedy policy is used instead of the greedy policy because it is a critical step in making the agent actually learn anything.
'''

import numpy as np
import tensorflow as tf
import cv2 # For rendering the environment

from dql_orbit_util import Object2D, simulate_gravity # ChatGPT-written object and function



# Helper function for reflecting vectors.
# Used in this project for calculating how balls bounce within a circular boundary.
def reflect_vector(vx:float, vy:float, rx:float, ry:float) -> tuple[float, float]:
    '''
    Reflect vector `(vx, vy)` across vector `(rx, ry)`.
    '''
    s = (rx*rx + ry*ry) ** -.5
    nx = ry * s
    ny = -rx * s
    d = 2 * (vx * nx + vy * ny)
    return vx - d * nx, vy - d * ny



class DodgeBallEnvironment:
    '''
    Not that "dodgeball" sport, but a simple environment where balls bounce around the screen and the agent must dodge them.
    
    It's a little bit like the Asteroids arcade game, but simpler because the only goal is to dodge, not to also try to aim and shoot.
    '''
    def __init__(self):
        self.stop = False # To tell when to stop rendering by pressing ESC.
        self.num_balls = 10
        self.time_scale = 0.03
        self.ball_min_radius = 0.05
        self.ball_max_radius = 0.1
        self.player_move_radius = 0.75
        self.player_move_radius_squared = self.player_move_radius * self.player_move_radius
        self.move_speed = 0.02
        self.reset()
    def move(self, x: float, y: float):
        self.pos += x * self.move_speed, y * self.move_speed
        
        # limit movement within player move radius
        ds = np.sum(np.square(self.pos))
        if ds >= self.player_move_radius_squared:
            self.pos *= self.player_move_radius * ds ** -.5
    def reset(self):
        self.pos = np.array([0., 0.])
        theta_pos = np.linspace(0, 2 * np.pi, num=self.num_balls, endpoint=False) + np.random.uniform(-.75, .75, size=self.num_balls)
        theta_vel = np.linspace(0, 2 * np.pi, num=self.num_balls, endpoint=False) + np.random.uniform(-.75, .75, size=self.num_balls) + np.pi
        r = np.random.uniform(self.ball_min_radius, self.ball_max_radius, size=self.num_balls)
        self.balls = np.column_stack((np.cos(theta_pos), np.sin(theta_pos), np.cos(theta_vel), np.sin(theta_vel), r, r*r))
        self.compute_player_ball_distances()
        self.terminal = False
    def step(self, action: int):
        if action == 0:
            # no movement
            pass
        elif action == 1:
            self.move(1, 0)
        elif action == 2:
            self.move(0, 1)
        elif action == 3:
            self.move(-1, 0)
        else:
            self.move(0, -1)
        
        # update ball positions
        self.balls[:, :2] += self.balls[:, 2:4] * self.time_scale
        
        # bounce balls off circular boundary (has radius of 1)
        ball_dist_squared = np.sum(np.square(self.balls[:, :2]), axis=1)
        for i, ds, (x, y, vx, vy, _, _) in zip(range(self.num_balls), ball_dist_squared, self.balls):
            if ds >= 1:
                angle = np.arccos(-(x*vx + y*vy) * (ds * (vx*vx + vy*vy)) ** -.5) # calculate angle between velocity and delta toward origin, utilizing the fact that dot(A, B) = ||A|| * ||B|| * cos(theta), where theta is the angle between the vectors A and B.
                if 2 * angle > np.pi: # just make sure that the angle is larger than pi/2 so that it doesn't get stuck in an infinite loop of bouncing outside of the boundary
                    # i.e. only bounce if the ball is heading away from the origin
                    self.balls[i, 2:4] = reflect_vector(-vx, -vy, x, y)

        self.compute_player_ball_distances()
        if np.any(self.player_ball_dist_squared <= np.square(self.balls[:, 4] + 3/64)):
            # The agent died!
            self.terminal = True
    def compute_player_ball_distances(self):
        self.player_ball_dist_squared = np.sum(np.square(self.balls[:, :2] - self.pos), axis=1)
    def get_reward(self) -> float:
        # Maximize reward by staying away from balls and staying near the origin.
        return np.min(self.player_ball_dist_squared) - np.sum(np.square(self.pos))
    def get_state(self) -> np.ndarray:
        balls = self.balls.copy()
        balls[:, :2] -= self.pos
        return np.concatenate((
            self.pos,
            balls[self.get_closest_balls_indices(num=6)].reshape((-1,)),
        ))
    def get_closest_balls_indices(self, num: int) -> list[int]:
        # Return the indices of the `num` closest balls to the player to be used as indices of DodgeBallEnvironment.balls.
        return list(sorted(range(self.num_balls), key=self.player_ball_dist_squared.__getitem__))[:num]
    def render(self):
        scale = 4
        img = np.zeros((128, 128, 3))
        img[:, :, 2] = 1 # red outside
        cv2.circle(img, (64, 64), 64, (0, 0, 0), -1) # black circle inside
        cv2.circle(img, (64, 64), int(64*self.player_move_radius), (.75, .75, .75), 0)
        img[int(64.5+64*self.pos[0]), int(64.5-64*self.pos[1])] = 0, 1, 0
        cv2.circle(img, (64, 64), int(64), (0, 0, 1), 0)
        for x, y, _, _, r, _ in self.balls:
            cv2.circle(img, (int(64.5+64*x), int(64.5-64*y)), int(64*r), (0, 0, 1), -1)
        img = np.repeat(np.repeat(img, scale, 0), scale, 1)
        cv2.imshow('env', img)
        inp = cv2.waitKey(1000//60)
        if inp & 0xFF == 27: # escape key
            self.stop = True




def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((38,)),
        tf.keras.layers.Dense(76),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(48),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(24),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(12),
        tf.keras.layers.LeakyReLU(0.1),
        tf.keras.layers.Dense(5),
    ])
    model.compile('adam', 'mse')
    return model



model = build_model()
env = DodgeBallEnvironment()

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
    if np.random.random() < epsilon:
        action = np.random.randint(5)
    else:
        action = np.argmax(model(np.expand_dims(old_state, 0)).numpy())

    env.step(action)

    new_state = env.get_state()
    reward = env.get_reward()
    terminal = env.terminal

    if (total_steps + 1) % force_reset_period == 0:
        terminal = True

    print(f'Epsilon: {epsilon:.2f} | Reward: {reward:+.2f} | Action: {action}')

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


