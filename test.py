from game_agent import Game_2048, Game_GUI
from agent import Agent
import torch

def test():
    total_score = 0
    
    game = Game_2048()
    gui = Game_GUI()
    agent = Agent(game)

    # Start from pretrained
    checkpoint = torch.load("models/Record_512_low_dim_rep.pth")
    agent.model.load_state_dict(checkpoint)

    while True:
        # Get a numerical representation of the state of the game
        state_old = agent.game_state(game.state)

        # Predict next move
        pred_move = agent.get_action(state_old)

        # Update the game based on the prediction
        score, reward, game_over = game.play_step(pred_move)

        # Visualize the game progress
        gui.render(game.state, game.score)

        if game_over:
            total_score += score

        if game.max_val == 2048:
            print("The AI wins")
            break

if __name__ == '__main__':
    test()