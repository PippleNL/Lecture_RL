import gym
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import pickle


def go_cart(link, max_steps, nb):
    pickle_model = pickle.load(open(link, 'rb'))

    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_steps
    nb_actions = env.action_space.n
    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(model=pickle_model, nb_actions=nb_actions, memory=memory)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.test(env, nb_episodes=nb, visualize=True)


go_cart('keras-rl/saved_models/model_test.obj', max_steps=2000, nb=10)
