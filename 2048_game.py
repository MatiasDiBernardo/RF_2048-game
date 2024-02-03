import pygame
import numpy as np
import sys
import random

class Game_2048():
    def __init__(self, font):
        self.state = self.set_initial_state()
        self.font = font
        self.score = 2
        self.max_val = 2
    
    def set_initial_state(self):
        state = np.zeros((4, 4))
        state[random.randint(0, 3), random.randint(0, 3)] = 2

        return state

    def dir_dicc(self, dir):
        if dir[0] == 1:
            return "LEFT"
        if dir[1] == 1:
            return "RIGHT"
        if dir[2] == 1:
            return "UP"
        if dir[3] == 1:
            return "DOWN"

    def update_squares(self, x_pos, y_pos, dir, value):
        move = True
        while(move):
            # I check for the cases where the block needs to move
            move = False 

            if dir == "LEFT" and x_pos != 0:
                x_pos -= 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos, x_pos + 1] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos, x_pos + 1] = 0
                    break

            if dir == "RIGHT" and x_pos != self.state.shape[0] - 1:
                x_pos += 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos, x_pos - 1] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos, x_pos - 1] = 0
                    break

            if dir == "DOWN" and y_pos != self.state.shape[0] - 1:
                y_pos += 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos - 1, x_pos] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos - 1, x_pos] = 0
                    break
                
            if dir == "UP" and y_pos != 0:
                y_pos -= 1
                # Check for emtpy space
                if self.state[y_pos, x_pos] == 0:
                    self.state[y_pos, x_pos] = value
                    self.state[y_pos + 1, x_pos] = 0
                    move = True
                    continue

                # Check for similar value to update
                if self.state[y_pos, x_pos] == value:
                    self.state[y_pos, x_pos] = value * 2
                    self.state[y_pos + 1, x_pos] = 0
                    break

    def update_state(self, move):

        direction = self.dir_dicc(move)
        print(direction)

        # Establish direction of the loop 
        n = self.state.shape[0]  #Only supports square tables
        if direction == "LEFT":
            iter = range(n)
        if direction == "RIGHT":
            iter = range(n - 1, -1, -1)
        if direction == "UP":
            iter = range(n)
        if direction == "DOWN":
            iter = range(n - 1, -1, -1)

        for i in iter:
            for j in iter:
                value = self.state[i, j]
                if value != 0:
                    self.update_squares(j, i, direction, value)
    
    def create_random_block(self):
        avaliable_indices = np.argwhere(self.state == 0)
        index_crate_random = avaliable_indices[random.randint(0, len(avaliable_indices) - 1)]
        self.state[index_crate_random[0], index_crate_random[1]] = 2
    
    def loss_game_condition(self):
        game_over = False
        # If the table is full 
        if np.all(self.state != 0):
            game_over = True
            movements = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            for mov in movements:
                old_state = np.copy(self.state)
                self.update_state(mov)
                # If a possible move exists the game continues
                if np.all(old_state != self.state):
                    game_over = False
        
        return game_over
    
    def calculate_reward(self, game_over):
        """It's rewarded when reaches a new max value
        on the table.

        Args:
            game_over (boolean): If the games ended is true.

        Returns:
            int: Reward value. (+10 if good, -10 if bad, 0 if neutral)
        """
        if game_over:
            return -10

        index_max = np.unravel_index(self.state.argmax(), self.state.shape)
        max_val_actual = self.state[index_max] 

        if max_val_actual > self.max_val:
            return 10
        
        return 0
    
    def play_step(self, move):
        if move is None:
            return 0, 0, False

        self.update_state(move)
        self.create_random_block()
        self.score = np.sum(self.state)
        game_over = self.loss_game_condition()
        reward = self.calculate_reward(game_over)

        print(self.state)
        print("      ")

        return self.state, reward, game_over
    
    
if __name__ == '__main__':

    WIDTH = 300
    HEIGTH = 500

    pygame.init()

    WIN = pygame.display.set_mode((WIDTH, HEIGTH))
    pygame.display.set_caption("My Pygame App")

    font = pygame.font.SysFont('arial', 25)
    game = Game_2048(font)
    print(game.state)

    while True:

        game_over = False
        move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move = [1, 0, 0, 0]

                if event.key == pygame.K_RIGHT:
                    move = [0, 1, 0, 0]

                if event.key == pygame.K_UP:
                    move = [0, 0, 1, 0]

                if event.key == pygame.K_DOWN:
                    move = [0, 0, 0, 1]

                state, reward, game_over = game.play_step(move)
        
        WIN.fill((0, 0, 0))
        pygame.display.update()

        if game_over:
            break