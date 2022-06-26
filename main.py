from email import policy
import gym
from environment import TetrisEnv
from stable_baselines3 import PPO
from os import path
from graph import graph
from multiprocessing import Process
import time
# import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Function for running graph.py and main.py simultaneously
def run_in_parallel(*fns):
  processes = []
  for fn in fns:
    process = Process(target=fn)
    process.start()
    processes.append(process)
  for process in processes:
    process.join()

def build_model(states, actions):

  input_tensor = Input(shape=(2, 22, 11))
  print(input_tensor)

  model = Sequential()
  model.add(Flatten(input_shape=(1, 22, 10)))
  model.add(Dense(6, activation='relu'))
  model.add(Dense(24, activation='relu'))
  model.add(Dense(actions, activation='linear'))

  return model

# def build_agent
def build_agent(model, actions):
  policy = BoltzmannQPolicy()
  memory = SequentialMemory(limit=50000, window_length=1)
  dqn = DQNAgent(model=model, nb_actions=actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)

  return dqn
    
# Function for training the model and running the game
def main():
  # Register Gym environment and create model
  print("Registering environment")
  gym.register('Tetris-v0', entry_point=TetrisEnv)
  env = gym.make('Tetris-v0')

  states = env.observation_space.shape
  actions = env.action_space.n

  print(states)
  print(actions)

  model = build_model(states, actions)

  print(model.summary())

  dqn = build_agent(model, actions)
  dqn.compile(Adam(lr=1e-3), metrics=['mae'])
  dqn.fit(env, nb_steps=10000, visualize=True, verbose=1)

  # model = PPO('MlpPolicy', env, verbose=1)

  # # Check if a model already exists and if so load it
  # print("Loading model")
  # if path.exists("trained_model.zip"):
  #     model.load("trained_model.zip")
  
  # # Learn for 20 000 steps
  # model.learn(20000)

  # # Save the model
  # print("Saving the model to trained_model.zip")
  # model.save("trained_model.zip")  

  # episodes = 10
  # # Run the game for 10 episodes
  # for episode in range(1, episodes+1):
  #   state = env.reset()
  #   done = False
  #   score = 0

  #   while not done:
  #     env.render()
  #     action = env.action_space.sample()
  #     n_state, reward, done, info = env.step(action)
  #     score += reward

  #   print("Episode {} finished. Score: {}".format(episode, score))

  
      

  # # Run the program infinitely
  # print("Running program")
  # env = gym.make('Tetris-v0')
  # print("Environment registered")
  # env.reset()
  # for i in range(50):
  #     env.render()
  #     env.step(env.action_space.sample()) # take a random action
  #     time.sleep(0.02)
  #     if i%10==0: print(i)
  # env.reset()
  # env.close()


if __name__ == "__main__":
    run_in_parallel(main, graph)     