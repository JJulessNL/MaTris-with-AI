import gym
from environment import TetrisEnv
from stable_baselines3 import PPO
from os import path
from graph import graph
from multiprocessing import Process
import time


# Function for running graph.py and main.py simultaneously
def run_in_parallel(*fns):
  processes = []
  for fn in fns:
    process = Process(target=fn)
    process.start()
    processes.append(process)
  for process in processes:
    process.join()

# Function for training the model and running the game
def main():
    # Register Gym environment and create model
    print("Registering environment")
    gym.register('Tetris-v0', entry_point=TetrisEnv)
    model = PPO('MlpPolicy', 'Tetris-v0')

    # Check if a model already exists and if so load it
    print("Loading model")
    if path.exists("trained_model.zip"):
        model.load("trained_model.zip")

    # Run the program infinitely
    print("Running program")
    env = gym.make('Tetris-v0')
    print("Environment registered")
    env.reset()
    for i in range(50):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        time.sleep(0.02)
        if i%10==0: print(i)
    env.reset()
    env.close()

    # while True:
    #     # Learn for 20 000 steps
    #     model.learn(20000)

    #     # Save the model
    #     print("Saving the model to trained_model.zip")
      #     model.save("trained_model.zip")  
      

if __name__ == "__main__":
    run_in_parallel(main, graph)     