"""
Title: pendulum_pygame
Author: [jadenhensley](https://github.com/jadenhensley)
Last modified: 2021/10/18
Description: Pendulum project, built using pygame and math modules.

Title: wheelPole
Author: [aimetz](https://github.com/aimetz)
Last modified: 2021/04/20

Title: gym/gym/envs/classic_control/pendulum.py
Author: [openai](https://github.com/openai)
Last modified: 2021/10/31
"""
import pygame
from math import pi, sin, cos
import numpy as np

import gym
from gym import spaces, logger


class Pendulum(gym.Env):
    def __init__(self, rend):

        self.theta_rod = 0
        self.theta_wheel = 0
        self.theta_rod_dot = 0
        self.theta_wheel_dot = 0
        self.len_rod = 0.5
        self.len_wheel = 0.5
        self.mass_rod = 20

        #self.rad_out = 0.05
        self.rad_out = 0.13# 0.105
        self.rad_in = 0.065
        #self.t = 0.01
        self.t = 0.03
        self.rho = 7870
        self.mass_wheel = 5.0295
        self.momentum_wheel = 0.0654
        #self.mass_wheel = (self.rad_out**2-self.rad_in**2)*self.t*pi*self.rho
        #self.momentum_wheel = self.mass_wheel*(self.rad_out**2+self.rad_in**2)/2

        #print(self.mass_wheel)
        #print(self.momentum_wheel)

        self.momentum_rod = 0.95
        self.dt = 0.005
        self.gravity = 9.81
        self.max_q1 = 5*pi/180 # stop training below this angle
        self.max_q1dot = 1 #initial q1_dot default 0.3? to be verified

        #self.wheel_max_speed = 20
        self.wheel_max_speed = 30 # not fixed
        #self.max_torque = 20
        self.max_torque = 21

        self.torque = 0
        self.last_torque = 0

        self.Ip = self.mass_rod*self.len_rod**2+self.mass_wheel*self.len_wheel**2+self.momentum_rod+self.momentum_wheel
        self.mbarg = (self.mass_rod*self.len_rod+self.mass_wheel*self.len_wheel)*self.gravity

        high = np.array([self.max_q1, self.max_q1dot, self.wheel_max_speed], dtype=np.float32)
        low = np.array([0, -self.max_q1dot, -self.wheel_max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        #print(self.mass_wheel)
        #print(self.momentum_rod)
        #print(self.momentum_wheel)

        width = 800
        height = 600
        self.origin_x = width//2
        self.origin_y = height//2
        self.POS = np.array([self.origin_x, self.origin_y])

        if rend:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Pendulum Simulation")
            pygame.font.init()
            self.debug_font = pygame.font.SysFont('Bauhuas 93', 30)
            self.hint_font = pygame.font.SysFont('Bauhaus 93', 26)
            #print("font")


    def reset(self, saved, curriculum_numerator=1000):
        # self.state is for render, self.agent_state is for training

        self.ang = 1*pi/180 # reset angle

        curriculum_denominator = 15
        if curriculum_numerator < curriculum_denominator:
            self.ang = self.ang * curriculum_numerator / curriculum_denominator

        if saved == None:
            reset_angle_random = np.random.uniform(low=-self.ang, high=self.ang)
            #reset_high = np.array([self.ang, self.max_q1dot, self.wheel_max_speed])
            #self.state = np.random.uniform(low=-reset_high, high=reset_high)
            self.state = np.array([reset_angle_random, 0, 0], dtype=np.float32)
            self.agent_state = np.array([abs(reset_angle_random), 0, 0], dtype=np.float32)
        else:
            self.state = np.array([self.ang, 0, 0], dtype=np.float32)
            self.agent_state = np.array([abs(self.ang), 0, 0], dtype=np.float32)
            # self.state = np.array([0, self.max_q1dot, 0],dtype=np.float32)

        #self.last_u = None
        self.last_torque = 0

        return self.agent_state


    def render(self, eval_run):
        #torque = action
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


        SCALE = 200
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        GRAY = (128, 128, 128)

        tip_x = self.POS[0]+self.len_wheel*sin(self.theta_rod)*SCALE
        tip_y = self.POS[1]-self.len_wheel*cos(self.theta_rod)*SCALE
        POSTIP = np.array([tip_x, tip_y])
        POSWHEEL = np.array(([tip_x+self.rad_out*sin(self.theta_wheel)*SCALE/4, tip_y-self.rad_out*cos(self.theta_wheel)*SCALE/4]))
        #print(POSTIP)
        self.screen.fill(WHITE)
        pygame.draw.line(self.screen, BLACK, self.POS, POSTIP, 10)
        pygame.draw.circle(self.screen, GRAY, POSTIP, self.rad_out*2*SCALE/4)
        pygame.draw.circle(self.screen, RED, POSWHEEL, self.rad_out*2*SCALE/5/4)
        img = self.hint_font.render("torque  : % .4f" %self.torque, True, BLACK)
        #img2 = self.hint_font.render("voltage: % .4f" %self.voltage, True, BLACK)
        img3 = self.hint_font.render("Evaluation Run %d" %eval_run, True, BLACK)
        self.screen.blit(img, (self.origin_x, self.origin_y/2-50))
        #self.screen.blit(img2, (self.origin_x, self.origin_y/2-30))
        self.screen.blit(img3, (self.origin_x/5, self.origin_y/2-50))

        pygame.display.update()


    def step(self, action):
        q1, q1_dot, q2_dot = self.state
        q1 = angle_normalize(q1)

        #q1 = self.theta_rod
        q2 = self.theta_wheel
        #q1_dot = self.theta_rod_dot
        #q2_dot = self.theta_wheel_dot
        #l1 = self.len_rod
        #l2 = self.len_wheel
        #m1 = self.mass_rod
        #m2 = self.mass_wheel
        #I1 = self.momentum_rod
        I2 = self.momentum_wheel
        dt = self.dt
        #g = self.gravity
        action_scale = self.max_torque

        if q1 >= 0:
            torque = action * action_scale
        else:
            torque = -action * action_scale
        torque = torque[0]

        #if q2_dot >= self.wheel_max_speed or q2_dot <= -self.wheel_max_speed:
        #    torque = 0

        Ip = self.Ip
        a = self.mbarg*sin(q1)

        q1_dot = q1_dot + ((a-torque)/(Ip-I2))*dt
        q1 = angle_normalize(q1 + q1_dot * dt)

        q2_dot = q2_dot + ((torque*Ip-a*I2)/I2/(Ip-I2))*dt
        #q2_dot = np.clip(q2_dot, -self.wheel_max_speed, self.wheel_max_speed)
        q2 = angle_normalize(angle_normalize(q2) + q2_dot * dt)

        #state = np.array([q1[0], q1_dot[0], q2_dot[0]], dtype=np.float32)
        self.state = (q1, q1_dot, q2_dot)
        if q1 >= 0:
            self.agent_state = self.state
        else:
            self.agent_state = (-self.state[0], -self.state[1], -self.state[2])

        done = bool(
            q1 < -self.max_q1
            or q1 > self.max_q1
        )
            #or q1_dot < -self.max_q1dot
            #or q1_dot > self.max_q1dot

        self.theta_rod = q1
        self.theta_wheel = q2
        self.theta_rod_dot = q1_dot
        self.theta_wheel_dot = q2_dot
        self.torque = torque

        # costs = 100 * q1 ** 2
        # costs = q1_dot ** 2
        # costs = 100 * q1 ** 2 + 1 * q1_dot ** 2
        costs = 100 * q1 ** 2 + 1 * q1_dot ** 2 + 0.0001 * (self.last_torque - torque) ** 2
        # costs = 100 * q1 ** 2 + 1 * q1_dot ** 2 + 0.0001 * torque ** 2
        # costs = 1000 * q1 ** 2 + 0.1 * q1_dot ** 2 + 0.001 * torque ** 2 + 0.00001 * q2_dot**2
        # costs = 100 * q1 ** 2 + 0.00001 * q2_dot ** 2
        # costs = 100 * q1 ** 2 + 1 * q1_dot ** 2 + 100 * torque ** 2 + 0.001 * q2_dot ** 2

        if done:
            costs += 1000

        #if abs(q1) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.1 :
        #    costs -= 1000
        #elif abs(q1) < 0.001 and abs(q1_dot) < 0.001 and abs(q2_dot) < 0.01 :
        #    costs -= 1000

        '''if q1 > 0.01 or q1 < -0.01 or q1_dot > 0.05 or q1_dot < -0.05:
            costs = 100 * q1 ** 2 + 1 * q1_dot ** 2 + 0.0001 * (self.last_torque - torque) ** 2
        else:
            costs = 0.000025 * q2_dot ** 2 + 0.0001 * (self.last_torque - torque) ** 2'''

        self.last_torque = torque

        #print("state: ", self.state)
        #print("agent state: ", self.agent_state)

        #return state, -costs, False, {}
        return np.array(self.agent_state, dtype=np.float32), -costs, done, {}

    def close(self):
        pygame.display.quit()
        pygame.quit()


def angle_normalize(th):
    return ((th+pi)%(2*pi))-pi