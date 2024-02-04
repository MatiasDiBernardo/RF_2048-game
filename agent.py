import torch
import random 
import numpy as np
import pygame
from collections import deque

from game_agent import Game_2048, Game_GUI

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent():
    def __init__(self, gui):
        self.game = Game_2048()
        self.gui = gui 

    def random_action(self, state):
        index = random.randint(0, 3)
        array = np.zeros(4)
        array[index] = 1
        return array
    
    def training(self):
        while True:
            state = self.game.state
            action = self.random_action(state)

            state, reward, game_over = self.game.play_step(action)
        
            # print("Los datos son: ")
            print("Num iter: ", self.game.iterations)
            # print("El reward: ", reward)
            # print("El score", self.game.score)
            # print("El estado del game", game_over)
            # print("---------------")

            if self.game.iterations == 10000:
                return self.game.max_val_list, self.game.max_score_list

            WIN.fill((0, 0, 0))
            self.gui.draw_score(self.game.score)
            self.gui.draw_blocks(self.game.state)
            pygame.display.update()

WIDTH = 310
HEIGTH = 400
pygame.init()

WIN = pygame.display.set_mode((WIDTH, HEIGTH))
font = pygame.font.SysFont('arial', 25)

gui = Game_GUI(WIN, font)
agent = Agent(gui)
list_values, list_score = agent.training()

print("El valor promedio con movimientos aleatories val max: ", np.sum(list_values)/len(list_values))
print("El valor promedio con score aleatories es: ", np.sum(list_score)/len(list_score))


