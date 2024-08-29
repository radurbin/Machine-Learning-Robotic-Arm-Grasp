import os, inspect
currentdir = os.getcwd()
parentdir = os.path.abspath(os.path.join(currentdir, os.pardir))
startdir = os.path.abspath(os.path.join(parentdir, os.pardir))
traydir = os.path.join(parentdir, "robot/table/table.urdf")
objectdir = os.path.join(parentdir, "3dmodels/poPdAb23/")
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from sawyer import sawyer
import time
import pybullet as p
import random
import pybullet_data
import pandas as pd
from random import seed
from random import randint
import statistics
import time
largeValObservation = 1
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
from math import *
from stable_baselines3.common.running_mean_std import RunningMeanStd
class sawyerEnv(gym.Env):

	metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
	def __init__(self, urdfRoot=pybullet_data.getDataPath(), 
		actionRepeat=1, 
		isEnableSelfCollision=True, 
		renders=False, 
		isDiscrete=False, 
		maxSteps=6000,
		graspType = "poPmAd35",
		orientation = 0,
		normOb= True, normReward=True, training = True, gamma=0.99):
		self.r = []
		self._isDiscrete = isDiscrete
		self._timeStep = 1. / 240.
		#self._timeStep = 0.01
		self._urdfRoot = urdfRoot
		self._actionRepeat = actionRepeat
		self._observation = []
		self._renders = renders
		self._maxSteps = maxSteps
		self._sawyerId = -1
		self.graspType = graspType
		self.orientation = orientation
		self.arm2hand = 0
		self._p = p
		self.num_envs = 1
		if self._renders:
			cid = p.connect(p.SHARED_MEMORY)
			if (cid < 0):
				cid = p.connect(p.GUI) #p.GUI to see it, p.DIRECT to let it run
			p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
		else:
			p.connect(p.DIRECT) 
		self.handPoint = 52   # "palm point" for each grasp type 
		#lowerObservation = [-5000.0, -5000.0, -5000.0, -5000.0, -5000.0, -5000.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -1.0, -1.0]
		#upperObservation = [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0]

		lowerObservation = [-3000]*6 + [-1]* 36
		upperObservation = [3000]*6 + [1]* 36
		#print("upperObservation", len(upperObservation))
		self.observation_space = spaces.Box(low=np.array(lowerObservation), high=np.array(upperObservation), dtype=np.float32)
		#print(self.observation_space)
		#observationDim = len(self.getExtendedObservation()) # 24
		#observation_high = np.array([largeValObservation] * observationDim)
		#self.observation_space = spaces.Box(-observation_high, observation_high)
		action_dim = 8
		self._action_bound = 1
		action_high = np.array([self._action_bound] * action_dim)
		self.action_space = spaces.Box(-action_high, action_high)
		self.viewer = None
		self.successGrasp = 0 # number of successfull grasp 
		self.evaluation = []
		self.height = -0.03
		self.seed()
		self.reset()

	def reset(self):
		self.terminated = 0
		p.resetSimulation()
		p.setPhysicsEngineParameter(numSolverIterations=150)
		p.setTimeStep(self._timeStep)
		#p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
		p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
		p.setGravity(0, 0, -10)
		self._sawyer = sawyer(timeStep=self._timeStep, graspType = self.graspType, orientation = self.orientation, handPoint = self.handPoint)
		self.trayUid = p.loadURDF(traydir, [1.22, 0.000000, -0.3], p.getQuaternionFromEuler([(math.pi/2), 0, (math.pi/2)]), useFixedBase = 1, flags = 8)
		########################################################################################################
		
		#TODO set self.index to different values to load different objects. The range is 0 to 7. Refer to “Object_assignment.pdf” for the randomly assigned object for each of you.
		self.index = 6
		
		########################################################################################################
		self.objectFeature = self.loadObject(self.graspType, self.index)
		self.object_position = [0.99, 0, -0.1]
		orn = p.getQuaternionFromEuler([self.objectFeature[5] * math.pi, self.objectFeature[6]* math.pi, self.objectFeature[7] * math.pi])
		#self.objectId = p.loadURDF(self.objectFeature[1], [self.objectFeature[9], self.objectFeature[10], self.objectFeature[11]], orn)
		self.objectId = p.loadURDF(self.objectFeature[1], self.object_position, orn)
		self._envStepCounter = 0
		p.stepSimulation()
		#self.realPos, self.realOrn = p.getBasePositionAndOrientation(self.objectId)
		self._observation = self.getExtendedObservation()
		self.episodeR = []
		self._graspSuccess = 0
		self.show = 0
		return np.array(self._observation)

	def r2r(self):
		# random 0 - 3
		seed(round(time.time()))
		return randint(0, 3)

	def loadObject(self, graspName, index):
		#i = self.r2r()
		#print(i)
		i = index
		#i = 4
		csvName = graspName + "_list.csv"		
		data = pd.read_csv(csvName)
		ob = data.iloc[i]['Object']
		l = data.iloc[i]['A'] * 0.01
		h = data.iloc[i]['B'] * 0.01
		w = data.iloc[i]['C'] * 0.01
		r = data.iloc[i]['Roll']
		p = data.iloc[i]['Pitch'] 
		y = data.iloc[i]['Yaw']
		shape = data.iloc[i]['Shape']
		objectPath = objectdir + ob + "/" + ob + ".urdf"
		return [ob, objectPath, l, h, w, r, p, y, shape]#, c_x, c_y, c_z]

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def __del__(self):
		p.disconnect()

	def getExtendedObservation(self):
		# all norm force
		palmForce = [] 
		thumbForce = []
		indexForce = []
		middleForce = []
		ringForce = []
		pinkyForce = []
		# all distance while norm force not zero	
		palmDist = []  
		thumbDist = []
		indexDist = []
		middleDist = []
		ringDist = []
		pinkyDist = []
		# define each part of the hand
		palmLinks = [19, 20, 25, 29, 34, 38, 43, 47, 52, 56, 57]
		thumbLinks = [58, 59, 60, 61, 62, 63, 64]
		indexLinks = [48, 49, 50, 51, 53, 54, 55]
		middleLinks = [39, 40, 41, 42, 44, 45, 46]
		ringLinks = [30, 31, 32, 33, 35, 36, 37]
		pinkyLinks = [21, 22, 23, 24, 26, 27, 28]

		# find contact point
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)

		# fill force and dist
		for i in range(nums):
			if(contact[i][3] in palmLinks):
				palmForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					palmDist.append(contact[i][8]) # contact distance
			
			if(contact[i][3] in thumbLinks):
				thumbForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					thumbDist.append(contact[i][8]) # contact distance
			
			if(contact[i][3] in indexLinks):	
				indexForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					indexDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in middleLinks):	
				middleForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					middleDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in ringLinks):
				ringForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					ringDist.append(contact[i][8]) # contact distance

			if(contact[i][3] in pinkyLinks):
				pinkyForce.append(contact[i][9]) # normal force
				if(contact[i][9] != 0):
					pinkyDist.append(contact[i][8]) # contact distance

		upperLimit = 1
		if(len(palmDist) != 0):
			palmd = min(palmDist)
		else:
			palmd = upperLimit
		if(len(thumbDist) != 0):
			td = min(thumbDist)
		else:
			td = upperLimit
		if(len(indexDist) != 0):
			ind = min(indexDist)
		else:
			ind = upperLimit
		if(len(middleDist) != 0):
			md = min(middleDist)
		else:
			md = upperLimit
		if(len(ringDist) != 0):
			rd = min(ringDist)
		else:
			rd = upperLimit
		if(len(pinkyDist) != 0):
			pind = min(pinkyDist)
		else:
			pind = upperLimit		
		dist = [palmd, td, ind, md, rd, pind]

		norm = [abs(sum(palmForce)), abs(sum(thumbForce)), abs(sum(indexForce)), abs(sum(middleForce)),  abs(sum(ringForce)), abs(sum(pinkyForce))] 
		#print([tcN, lcN])
		handState = p.getLinkState(self._sawyer.sawyerId, self.handPoint)
		handPos = handState[0]
		handOrn = handState[1]
		#print(handPos)
		obPos, obOrn = p.getBasePositionAndOrientation(self.objectId)
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)	
		midTip = p.getLinkState(self._sawyer.sawyerId, 42)	
		ringTip = p.getLinkState(self._sawyer.sawyerId, 33)	
		pinkyTip = p.getLinkState(self._sawyer.sawyerId, 24)	
		obHand = self.relativePos(handPos, handOrn, obPos, obOrn)
		obThumb = self.relativePos(thumbTip[0], thumbTip[1], obPos, obOrn)
		obIndex = self.relativePos(indexTip[0], indexTip[1], obPos, obOrn)
		obMid = self.relativePos(midTip[0], midTip[1], obPos, obOrn)
		obRing = self.relativePos(ringTip[0], ringTip[1], obPos, obOrn)
		obPinky = self.relativePos(pinkyTip[0], pinkyTip[1], obPos, obOrn)
		self._observation = norm + dist + obHand + obThumb + obIndex + obMid + [self.objectFeature[8]] + [p.getClosestPoints(self._sawyer.sawyerId, self.objectId, 500, self.handPoint, -1)[0][8]] + [self.objectFeature[2], self.objectFeature[3], self.objectFeature[4]] + [obPos[2]]
		return np.array(self._observation)

	def relativePos(self, handPos, handOrn, obPos, obOrn):
		invhandPos, invhandOrn = p.invertTransform(handPos, handOrn)
		handEul = p.getEulerFromQuaternion(handOrn)    
		obPosInHand, obOrnInHand = p.multiplyTransforms(invhandPos, invhandOrn, obPos, obOrn)
		projectedObPos2D = [obPosInHand[0], obPosInHand[1]]
		obEulerInHand = p.getEulerFromQuaternion(obOrnInHand)
		obInHandPosXYEulZ = [obPosInHand[0], obPosInHand[1], obPosInHand[2], obEulerInHand[0], obEulerInHand[1], obEulerInHand[2]]
		return obInHandPosXYEulZ

	# find which parts have contact with the object 
	# if a part provides norm force, we count it as 
        # a contact part. 1 for contact, 0 for no contact 
	def getContactPart(self):		
		contactParts = [0, 0, 0, 0, 0, 0] # palm, thumb, index, middle, ring, pink		
		# define each part of the hand
		palmLinks = [19, 20, 25, 29, 34, 38, 43, 47, 52, 56, 57]
		thumbLinks = [58, 59, 60, 61, 62, 63, 64]
		indexLinks = [48, 49, 50, 51, 53, 54, 55]
		middleLinks = [39, 40, 41, 42, 44, 45, 46]
		ringLinks = [30, 31, 32, 33, 35, 36, 37]
		pinkyLinks = [21, 22, 23, 24, 26, 27, 28]
		arm2handLinks = [19, 20, 21, 25, 26, 29, 30, 34, 35, 38, 39, 43, 44, 47, 48, 52, 53, 56, 57]

		# find contact point
		contact = p.getContactPoints(self._sawyer.sawyerId, self.objectId)
		nums = len(contact)
		# fill force and dist
		limitForce = 1
		for i in range(nums):
			if(contact[i][3] in palmLinks):
				if(contact[i][9] >= limitForce):
					contactParts[0] = 1
			
			if(contact[i][3] in thumbLinks):
				if(contact[i][9] >= limitForce):
					contactParts[1] = 1
			
			if(contact[i][3] in indexLinks):
				if(contact[i][9] >= limitForce):
					contactParts[2] = 1

			if(contact[i][3] in middleLinks):
				if(contact[i][9] >= limitForce):
					contactParts[3] = 1

			if(contact[i][3] in ringLinks):
				if(contact[i][9] >= limitForce):
					contactParts[4] = 1

			if(contact[i][3] in pinkyLinks):
				if(contact[i][9] >= limitForce):
					contactParts[5] = 1
			if(contact[i][3] in arm2handLinks):
				self.arm2hand = 1

		return contactParts

	def step(self, action):
		#print("action: ", action)
		#print(action)
		#self.st = p.getClosestPoints(self._sawyer.sawyerId, self.objectId, 500, self.handPoint, -1)[0][8] # stage switch trigger: dist object and mid
		#print("self.st: ", self.st)	
		d1 = 0.02 # stage 1 scaler
		d2 = 0.001 # stage 2 scaler 
		d3 = 1.5 # finger move scaler
		
		if(self.inPosition()): # stage 2
			dx = action[0] * d2
			dy = action[1] * d2
			dz = action[2] * d2
			da1 = action[3] * 0.4
			da2 = action[4] * d3
			da3 = action[5] * d3 * 0.8
			da4 = action[6] * d3
			da5 = action[7] * d3
			realAction = [dx, dy, dz, da1, da2, da3, da4, da5]
		else: # stage 1
			dx = action[0] * d1
			dy = action[1] * d1
			dz = action[2] * d1
			realAction = [dx, dy, dz, 0, 0, 0, 0, 0]
		return self.step1(realAction)

	def step1(self, action):
		for i in range(self._actionRepeat):
			self._sawyer.applyAction(action, self.terminated)
			p.stepSimulation()
			if self._termination(action):
				break
			self._envStepCounter += 1
			if self._renders:
				time.sleep(self._timeStep)
		reward = self._reward()
		self._observation = self.getExtendedObservation()
		contactParts = self.getContactPart()
		if(contactParts[3]):
			reward = reward + 500

		if (self._graspSuccess):
			reward = reward + 100000
			self.successGrasp = self.successGrasp + 1
		#debug = self.successGrasp
		debug = {'grasp_success': self._graspSuccess}
		done = self._termination(action)
		self.episodeR.append(reward)
		#print("Obs", self._observation)
		if done:
			self.episodeR.append(self._graspSuccess)
			self.evaluation.append(self.episodeR)
			#print("reward = ", reward)
		#print(self._observation)
		#print(len(self._observation))
		return self._observation, reward, done, debug


	def render(self):
		return 0

	def _termination(self, action):	
		if (self.terminated or self._envStepCounter > self._maxSteps):
			self._observation = self.getExtendedObservation()
			#print("stop due to time out")
			return True
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		contactParts = self.getContactPart()
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		palmTip = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		
		if((obPos[0]> self.object_position[0] + 0.04) or (obPos[0] < self.object_position[0] - 0.04)):
			#if( not ((obPos[0] + self.objectFeature[4]*0.5 + 0.006 < indexTip[0]) and (thumbTip[0] < obPos[0] - self.objectFeature[4]*0.5 - 0.006))):
			if( not ((obPos[0] + self.objectFeature[4]*0.5 < indexTip[0]) and (thumbTip[0] < obPos[0] - self.objectFeature[4]*0.5))):
				self._observation = self.getExtendedObservation()
				#print("ObjectNum: ", self.objectFeature[0])
				#print("Terminated: x out of range")
				time.sleep(1)
				return True
		if((obPos[1]> self.object_position[1] + 0.04) or (obPos[1] < self.object_position[1]-0.04)):
			if((palmTip[1] >= obPos[1] +  self.objectFeature[2]*0.5) or  (palmTip[1] <= obPos[1] - self.objectFeature[2]*0.5)):
				self._observation = self.getExtendedObservation()
				#print("ObjectNum: ", self.objectFeature[0])
				#print("Terminated: y out of range")
				time.sleep(1)
				return True
		
		if(contactParts[1] and contactParts[2] and self.inPosition()): 
			self.terminated = 1
			for i in range(200):
				self._sawyer.applyAction(action, self.terminated)
				p.stepSimulation()
				objectPosCurrent = p.getBasePositionAndOrientation(self.objectId)[0]
				if (objectPosCurrent[2] > self.height):
					#print("ObjectNum: ", self.objectFeature[0])
					#print("st:", self.st)
					self._graspSuccess = 1
					#print("Terminated: successfully grasp")
					time.sleep(1)
					break
			self._observation = self.getExtendedObservation()
			if(not self._graspSuccess):
				#print("Terminated: Object slipped")
				return True
		return False
	
	def _reward(self): 

		reward_s1 = self.reward_s1()
		reward_s2 = self.reward_s2()
		#print(self.inPosition())
		if(self.inPosition()): # stage 2
			reward = 2000 + 1.05*reward_s1 + reward_s2
			#reward = transition + reward_s2
			#print("stage 2: ", reward)		
		else:# stage 1
			reward = reward_s1  
			#print("stage 1: ", reward)
		return reward


	#TODO develop the reward_s1 function to have to robot arm approaching to the grasp location
	def reward_s1(self):	
		reward = 0
		###################### some parameters MAYBE useful ####################################################		
		obPos, _ = p.getBasePositionAndOrientation(self.objectId) # the position of the object
		handPos = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0] # the current position of the hand			
		dist = p.getClosestPoints(self.objectId, self._sawyer.sawyerId, 500, -1, self.handPoint)[0][8] # the distance between the hand and the object
		x_rdy = self.xInRange() # a boolean value that indicates the hand is in positon in x axis
		y_rdy = self.yInRange() # a boolean value that indicates the hand is in positon in y axis
		z_rdy = self.zInRange() # a boolean value that indicates the hand is in positon in z axis		
		
		###################### implement your function below ####################################################
		
		# THIS REWARD FUNCTION GOT A 78% SUCCESS RATE
		# goal is to first move the arm above the object, then lower the arm down to the object
		# reward the robot if the distance is within some distance
		distance_threshold_over = 0.3
		# reward the robot if the distance is within a closer distance and directly above the object (x and y are aligned)
		distance_threshold_close = 0.1
		# reward to give the robot if within that first distance
		reward_for_being_over = 10
		# bigger reward if the robot is closer and aligned in the x and y
		reward_for_being_close = 100
		# if the arm is within the first distance threshold, reward it, rewarding it more for being closer to the object
		if dist < distance_threshold_over:
			reward += reward_for_being_over * (1 - (dist / distance_threshold_over)) # second part of this should give it higher rewards for being closer
		# if the arm is within the second distance threshold and it is directly above it, add to the reward, rewarding it more for being closer
		if dist < distance_threshold_close and x_rdy and y_rdy:
			reward += reward_for_being_close * (1 - (dist / distance_threshold_close))
		# add a huge final reward if the arm is fully aligned (should grasp here)
		if x_rdy and y_rdy and z_rdy:
			reward += 500

		return reward

	def reward_s2(self):
		#reward = 0
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		#reward = 0.05 * self.reward_s1()
		contactParts = self.getContactPart()
		#if(sum(contactParts) > 0):
		x = sum(contactParts)	
		reward = (x-1)*150 # minus 2 since at least 2 contact parts are required		
		return reward

	def xInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		return (obPos[0] + self.objectFeature[4]*0.5 + 0.004 < indexTip[0]) and (obPos[0] - self.objectFeature[4]*0.5 - 0.004> thumbTip[0])
	
	def yInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		palmTip = p.getLinkState(self._sawyer.sawyerId, self.handPoint)[0]
		#return (palmTip[1] < obPos[1] + self.objectFeature[2]*0.25) and (palmTip[1] > obPos[1] - self.objectFeature[2]*0.25) 
		return (palmTip[1] < obPos[1] + 0.02) and (palmTip[1] > obPos[1] - 0.02) 
	'''
	def yInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		return (indexTip[1] < obPos[1] + self.objectFeature[2]*0.2) and (indexTip[1] > obPos[1] - self.objectFeature[2]*0.2)
	'''
	def zInRange(self):
		obPos, _ = p.getBasePositionAndOrientation(self.objectId)
		thumbTip = p.getLinkState(self._sawyer.sawyerId, 62)[0]
		indexTip = p.getLinkState(self._sawyer.sawyerId, 51)[0]
		#upper = (obPos[2]+ 0.49*self.objectFeature[3] > thumbTip[2]) and (obPos[2]+0.49*self.objectFeature[3] > indexTip[2])
		#lower = (obPos[2]- 0.4*self.objectFeature[3] < thumbTip[2]) and (obPos[2]-0.4*self.objectFeature[3] < indexTip[2])
		upper = (obPos[2]+0.5*self.objectFeature[3] > indexTip[2]) and obPos[2]+0.5*self.objectFeature[3] > thumbTip[2]
		#lower = obPos[2]-0.4*self.objectFeature[3] < indexTip[2]
		return upper 


	def inPosition(self):
		return self.xInRange() and self.yInRange() and self.zInRange()

	def distant(self, a, b):
		return sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

	def eva(self):
		return self.evaluation

	def eMean(self):
		m = []
		for i in range(len(self.evaluation)):
			m.append(statistics.mean(self.evaluation[i]))
		return 	statistics.mean(m)

	def handReading(self):
		return self._sawyer.handReading	

	def sus(self):
		return self._graspSuccess

	def o2o(self):
		# one by one
		i = objectIndex
		return i%4

	def m2o(self, n):
		if(self.objectIndex < n):
		    i = 0 
		elif(self.objectIndex in range(n,2*n)):
		    i = 1 
		elif(self.objectIndex in range(2*n,3*n)):
		    i=2
		elif(self.objectIndex in range(3*n,4*n)):
		    i=3
		else:
		    i = self.objectIndex%4
		return i



