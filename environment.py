import numpy as np
import pygame
from matris import HEIGHT, MATRIX_HEIGHT, MATRIX_WIDTH, WIDTH, Game, Matris
import gym
from gym import spaces, Env
from pynput.keyboard import Key, Controller

class TetrisEnv(Env):
    def __init__(self):
        super().__init__()

        # Create an actionspace with 6 actions (nothing, rotate, left, right, soft drop, hard drop)
        self.action_space = spaces.Discrete(6)

        # Create a observation space with the board data, 0 for empty, 1 untill 7 for filled, depending on blocktype
        self.observation_space = spaces.Box(low=0, high=1, shape=(MATRIX_HEIGHT, MATRIX_WIDTH))

        # Initialize pynput controller used for simulating keypresses
        self.keyboard = Controller()

        # Initialize the game
        pygame.init()

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MaTris")
        self.game = Game()
        self.game.main(screen)
        print("GAME INITIALIZED")

    def step(self, action: int):
        print("STEP: " + str(self.step_counter))	
        self._give_input(action)

        gamestate = self.game.matris.receive_gamestate()

        # Get observation
        observation = gamestate['matrix']
        data = list(observation.items())
        observation_array = np.array(data)
        observation_matrix = np.empty((MATRIX_HEIGHT, MATRIX_WIDTH))
        k = 0

        for i in range(MATRIX_HEIGHT):
            for j in range(MATRIX_WIDTH):
                if(type(observation_array[k][1]) == tuple):
                    observation_matrix[i][j] = 1
                else:
                    observation_matrix[i][j] = observation_array[k][1]
                k += 1
        

        # Get reward
        reward = self._get_reward(gamestate, action)
        print(gamestate["isAlive"])

        info = {}
        timepassed = self.game.clock.tick(50)
        if self.game.matris.update((timepassed / 1000.) if not self.game.matris.paused else 0):
            gamestate = self.game.matris.receive_gamestate()
            if gamestate["isAlive"]:
                print("Alive: " +  str(gamestate["isAlive"]))
                self.game.redraw()


        # Check if the game is over
        if gamestate["isAlive"] == False:
            # Print total reward
            print("Reward is", self.total_reward)

            # Log total reward
            with open('rewards.txt', 'a', encoding='utf-8') as file:
                file.write(str(self.total_reward["total"]) + " " + str(self.score) + "\n")
            done = True
        else:
            done = False

        # Increase step counter
        self.step_counter += 1

        return observation_array, reward, done, info

    # Input function
    def _give_input(self, action: int):
        # Reset last input
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.down)
        self.keyboard.release(Key.right)
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.space)

        # Do nothing
        if action == 0:
            pass
        # Rotate
        elif action == 1:
            self.keyboard.press(Key.up)
        # Move left
        elif action == 2:
            self.keyboard.press(Key.left)
        # Move right
        elif action == 3:
            self.keyboard.press(Key.right)
        # Hard drop
        elif action == 4:
            self.keyboard.press(Key.space)
        # Soft drop
        elif action == 5:
            self.keyboard.press(Key.down)

    # Possible rewards and penalties
    def _get_reward(self, gamestate: dict, action:int):
        reward = 0

        # Calculating the reward for the action
        reward += (gamestate['score']- self.score)

        # Saving this score also in total_reward
        self.total_reward["score"] += (gamestate["score"] - self.score)

        # Saving the current score in self.score, to use it as the old score in the next iteration.
        self.score = gamestate["score"]

        # Passing of time gives a reward (surive longer)
        reward += 5
        self.total_reward["time_alive"] += 5

        # Pressing buttons is not free
        if action != 0 or action != 4:
            reward -= 5
            self.total_reward["button_presses"] -= 5

       # Dying gives a penalty
        if gamestate["isAlive"] == False:
            reward -= 500
            self.total_reward["dying"] -= 500

        # Saving the total reward in self.total_reward for this step
        self.total_reward["total"] += reward 

        return reward 

    def render(self, mode='human', close=False):
        return None

    def reset(self):
        # Reset total_reward variable (used for logging) to 0
        self.total_reward = {
            "score": 0,
            "time_alive": 0,
            "button_presses": 0,
            "dying": 0,
            "total": 0
        }

        # Reset score variable
        self.score = 0

        # Reset step counter
        self.step_counter = 0

        # Reset the game
        self.game.restartGame()

        # Set input to none
        self._give_input(0)

        # Read an observation
        gamestate = self.game.matris.receive_gamestate()
        observation = gamestate['matrix']
        data = list(observation.items())
        observation_array = np.array(data)
        observation_matrix = np.empty((MATRIX_HEIGHT, MATRIX_WIDTH))
        k = 0

        for i in range(MATRIX_HEIGHT):
            for j in range(MATRIX_WIDTH):
                if(type(observation_array[k][1]) == tuple):
                    observation_matrix[i][j] = 1
                else:
                    observation_matrix[i][j] = observation_array[k][1]
                k += 1

        print(observation_matrix)
        print("RESETTING")
        # Return the observation
        return observation_matrix

                