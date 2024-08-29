import os
import gym
import time
import numpy as np
from datetime import datetime
from sawyerEnv import sawyerEnv
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG, DQN, SAC
from typing import Callable
#from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import load_results, ts2xy
from Monitor import Monitor
import pandas as pd
from successRateCallBack import successRateCallBack
################################################# Define Variables ########################################################################
n_envs = 1
orientation = 1 # 0 from side, 1 from above, 2 from above 1
graspType = "poPdAb23"
log_dir = "log"
fileName = log_dir + "/episodeData"
modelName = log_dir + "/" + graspType
envName = log_dir + "/" + graspType + ".pkl"
################################################# Training and Evaluation #################################################################
# create environment and custom callback
env = sawyerEnv(renders=True, isDiscrete=False, maxSteps=1024, graspType = graspType, orientation = orientation)
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_reward=True) # when training norm_reward = True 

############################################ train the model ##############################################################

#TODO build a RL learning algorithm to train the model. 
# use in stable_baselines3
# parameter env is the training environment

timeStep = 20480 # decide the length of the training, originally 20480
# I left this as the timestep and it had better results than when I changed it
model = PPO("MlpPolicy", env, verbose=1)
# using PPO as the RL learning algorithm
# "PPO("MlpPolicy", env, verbose=1)" is from the stable_baselines3 documentation
model.learn(total_timesteps=timeStep)
######################################################################################################################################

# save model
model.save(modelName)
env.save(envName)
env.close()



