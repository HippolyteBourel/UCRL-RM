import numpy as np
import sys
from six import StringIO, b

import scipy.stats as stat
import matplotlib.pyplot as plt

from gym import utils
from gym.envs.toy_text import discrete
import environments.discreteMDP

from gym import Env, spaces
import string

from environments import gridworld, rewardMachine, discreteMDP
#from environments import MDPRM_utils

def categorical_sample(prob_n, np_random):
	"""
	Sample from categorical distribution
	Each row specifies class probabilities
	"""
	prob_n = np.asarray(prob_n)
	csprob_n = np.cumsum(prob_n)
	return (csprob_n > np_random.rand()).argmax()

def clip(x,range):
	return max(min(x,range[1]),range[0])


def fourRoom(X,Y):
	Y2 = (int) (Y/2)
	X2 = (int) (X/2)
	maze = np.ones((X,Y))
	for x in range(X):
		maze[x][0] = 0.
		maze[x][Y-1] = 0.
		maze[x][Y2] = 0.
	for y in range(Y):
		maze[0][y] = 0.
		maze[X-1][y] = 0.
		maze[X2][y] = 0.
		maze[X2][(int) (Y2/2)] = 1.
		maze[X2][(int) (3*Y2/2)] = 1.
		maze[(int) (X2/2)][Y2] = 1.
		maze[(int) (3*X2/2)][Y2] = 1.
	return maze

def RM_fourRoom_patrol8(S, X, Y):
	events = np.array([[[None for _ in range(S)] for _ in range(4)] for _ in range(S)])
	for s in range(S):
		for a in range(4):
			if s != 0:
				events[0, a, s] = 0
			if s != S - 1: 
				events[S - 1, a, s] = 1
	transitions = np.array([[1, 0], [1, 0]])
	rewards = np.array([[1, 0], [0, 0]])
	return events, transitions, rewards


def twoRoom(X,Y):
	X2 = (int) (X/2)
	maze = np.ones((X,Y))
	for x in range(X):
		maze[x][0] = 0.
		maze[x][Y-1] = 0.
	for y in range(Y):
		maze[0][y] = 0.
		maze[X-1][y] = 0.
		maze[X2][y] = 0.
	maze[X2][ (int) (Y/2)] = 1.
	return maze


def mapping_2room(x, y, Y):
	return y*(Y) + x

def RM_twoRoom_4corner(nX, nY): # 0 = A, 1 = B1, 2 = B2, 3 = C1, 4 = C2, 5 = D1, 6 = D2, 7 = E1, 8 = E2
	X = nX - 2
	Y = nY - 2
	X2 = (int) (X/2)
	sA = mapping_2room(X2, 1, Y)
	sB1 = X2
	sB2 = 0
	sC1 = mapping_2room(X2 - 1, 1, Y)
	sC2 = X - 1
	sD1 = mapping_2room(X2, 2, Y)
	sD2 = mapping_2room(0, Y - 1, Y)
	sE1 = mapping_2room(X2 + 1, 1, Y)
	sE2 = mapping_2room(X - 1, Y - 1, Y)
	events = np.array([[None for _ in range(4)] for _ in range(nX * nY)])
	for a in range(4):
		events[sA, a] = 0
		events[sB1, a] = 1
		events[sB2, a] = 2
		events[sC1, a] = 3
		events[sC2, a] = 4
		events[sD1, a] = 5
		events[sD2, a] = 6
		events[sE1, a] = 7
		events[sE2, a] = 8
	transitions = np.array([[0, 1, 0, 3, 0, 5, 0, 7, 0], 
							[1, 1, 2, 1, 1, 1, 1, 1, 1],
							[0, 2, 2, 2, 2, 2, 2, 2, 2],
							[3, 3, 3, 3, 4, 3, 3, 3 ,3],
							[0, 4, 4, 4, 4, 4, 4, 4, 4],
							[5, 5, 5, 5, 5, 5, 6, 5, 5],
							[0, 6, 6, 6, 6, 6, 6, 6, 6],
							[7, 7, 7, 7, 7, 7, 7, 7, 8],
							[0, 8, 8, 8, 8, 8, 8, 8, 8]])
	max_i = 9
	rewards = np.zeros((max_i, max_i))
	for i in range(max_i):
		rewards[0, i] = 1
	return events, transitions, rewards

def RM_twoRoom_3corner(nX, nY): # 0 = A, 1 = B1, 2 = B2, 3 = C1, 4 = C2, 5 = D1, 6 = D2
	X = nX - 2
	Y = nY - 2
	X2 = (int) (X/2)
	sA = mapping_2room(X2, 1, Y)
	sB1 = X2
	sB2 = 0
	sC1 = mapping_2room(X2 - 1, 1, Y)
	sC2 = X - 1
	sD1 = mapping_2room(X2, 2, Y)
	sD2 = mapping_2room(0, Y - 1, Y)
	events = np.array([[None for _ in range(4)] for _ in range(nX * nY)])
	for a in range(4):
		events[sA, a] = 0
		events[sB1, a] = 1
		events[sB2, a] = 2
		events[sC1, a] = 3
		events[sC2, a] = 4
		events[sD1, a] = 5
		events[sD2, a] = 6
	transitions = np.array([[0, 1, 0, 3, 0, 5, 0], 
							[1, 1, 2, 1, 1, 1, 1],
							[0, 2, 2, 2, 2, 2, 2],
							[3, 3, 3, 3, 4, 3, 3],
							[0, 4, 4, 4, 4, 4, 4],
							[5, 5, 5, 5, 5, 5, 6],
							[0, 6, 6, 6, 6, 6, 6]])
	max_i = 7
	rewards = np.zeros((max_i, max_i))
	for i in range(max_i):
		rewards[0, i] = 1
	return events, transitions, rewards

def RM_twoRoom_2corner(nX, nY): # 0 = A, 1 = B1, 2 = B2, 3 = C1, 4 = C2
	X = nX - 2
	Y = nY - 2
	X2 = (int) (X/2)
	sA = mapping_2room(X2, 1, Y)
	sB1 = X2
	sB2 = 0
	sC1 = mapping_2room(X2 - 1, 1, Y)
	sC2 = X - 1
	events = np.array([[None for _ in range(4)] for _ in range(nX * nY)])
	for a in range(4):
		events[sA, a] = 0
		events[sB1, a] = 1
		events[sB2, a] = 2
		events[sC1, a] = 3
		events[sC2, a] = 4
	transitions = np.array([[0, 1, 0, 3, 0], 
							[1, 1, 2, 1, 1],
							[0, 2, 2, 2, 2],
							[3, 3, 3, 3, 4],
							[0, 4, 4, 4, 4]])
	max_i = 5
	rewards = np.zeros((max_i, max_i))
	for i in range(max_i):
		rewards[0, i] = 1
	return events, transitions, rewards

def RM_twoRoom_1corner(nX, nY): # 0 = A, 1 = B1, 2 = B2
	X = nX - 2
	Y = nY - 2
	X2 = (int) (X/2)
	sA = mapping_2room(X2, 1, Y)
	sB1 = X2
	sB2 = 0
	events = np.array([[None for _ in range(4)] for _ in range(nX * nY)])
	for a in range(4):
		events[sA, a] = 0
		events[sB1, a] = 1
		events[sB2, a] = 2
	transitions = np.array([[0, 1, 0], 
							[1, 1, 2],
							[0, 2, 2]])
	max_i = 3
	rewards = np.zeros((max_i, max_i))
	rewards[3, 0]
	return events, transitions, rewards



def RM_twoRoom_patrol2(S):
	events = np.array([[[None for _ in range(S)] for _ in range(4)] for _ in range(S)])
	for s in range(S):
		for a in range(4):
			if s != 0:
				events[0, a, s] = 0
			if s != S - 1: 
				events[S - 1, a, s] = 1
	transitions = np.array([[1, 0], [1, 0]])
	rewards = np.array([[1, 0], [0, 0]])
	return events, transitions, rewards

def RM_riverSwim_patrol2_sas(S):
	events = np.array([[[None for _ in range(S)] for _ in range(2)] for _ in range(S)])
	events[1, 0, 0] = 0
	events[1, 1, 0] = 0
	events[S - 2, 0, S - 1] = 1
	events[S - 2, 1, S - 1] = 1
	transitions = np.array([[0, 1], [0, 1]])
	rewards = np.array([[0, 1], [0, 0]])
	return events, transitions, rewards
	



def RM_riverSwim_patrol2(S):
	events = np.array([[None, None] for _ in range(S)])
	events[0, 0] = 0
	events[0, 1] = 0
	events[S - 1, 0] = 1
	events[S - 1, 1] = 1
	transitions = np.array([[0, 1], [0, 1]])
	rewards = np.array([[0, 1], [0, 0]])
	return events, transitions, rewards



def RM_riverSwim_patrol2_s(S):
	events = np.array([None for _ in range(S)])
	events[0] = 0
	events[0] = 0
	events[S - 1] = 1
	events[S - 1] = 1
	transitions = np.array([[0, 1], [0, 1]])
	rewards = np.array([[0, 1], [0, 0]])
	return events, transitions, rewards



def RM_Flower(sizeB):
	liste_1_step = [1, 2, 4, 5, 7, 8, 10, 11]
	liste_back_center = [3, 6, 9, 12]
	events = np.array([[None, None] for _ in range(6)])
	events[1, 0] = 0 #small 1
	events[2, 0] = 1 #small 2
	events[3, 0] = 2 #small 3
	events[4, 0] = 3 #small 4
	events[5, 0] = 4 #Big loop
	transitions = []
	nQ = 1 + 4*3 + sizeB #central state, 4 small loop, one big loop
	rewards = []
	for q in range(nQ):
		reward = [0 for _ in range(5)]
		if q == 0:
			transition = [0 for _ in range(5)]
			transition[0] = 1 #start small loop 1
			transition[1] = 4 #start small loop 2
			transition[2] = 7 #start small loop 3
			transition[3] = 10 #start small loop 4
			transition[4] = 13 #start big loop
		# small loop 1
		elif q in [1, 2]:
			transition = [q for _ in range(5)]
			transition[0] = q + 1
		elif q == 3:
			transition = [q for _ in range(5)]
			transition[0] = 0
			reward[0] = 1
		# small loop 2
		elif q in [4, 5]:
			transition = [q for _ in range(5)]
			transition[1] = q + 1
		elif q == 6:
			transition = [q for _ in range(5)]
			transition[1] = 0
			reward[1] = 1
		# small loop 3
		elif q in [7, 8]:
			transition = [q for _ in range(5)]
			transition[2] = q + 1
		elif q == 8:
			transition = [q for _ in range(5)]
			transition[2] = 0
			reward[2] = 1
		# small loop 4
		elif q in [10, 11]:
			transition = [q for _ in range(5)]
			transition[3] = q + 1
		elif q == 12:
			transition = [q for _ in range(5)]
			transition[3] = 0
			reward[3] = 1
		# BIG loop
		elif q < nQ - 1:
			transition = [q for _ in range(5)]
			transition[4] = q + 1
		else: #nQ - 1
			transition = [q for _ in range(5)]
			transition[4] = 0
			reward[4] = 1
		transitions.append(transition)
		rewards.append(reward)
	transitions = np.array(transitions)
	rewards = np.array(rewards)
	return events, transitions, rewards


class GridWorld_RM(environments.gridworld.GridWorld):
	metadata = {'render.modes': ['text', 'ansi', 'pylab'], 'maps': ['2-room RM']}

	def __init__(self, sizeX, sizeY, map_name="2-room_patrol2", slippery=0.1,nbGoals=1,rewardStd=0.,density=0.2, lengthofwalks=5, initialSingleStateDistribution=False):
		# initialSingleStateDistribution: If set to True, the initial distribution is a Dirac at one state (this state is uniformly chosen amongts valid non-goal states)
		# If set to False, then the initial distribution is set to be uniform over all valid non-goal states.
		
		#desc = maps[map_name]
		self.sizeX, self.sizeY = sizeX, sizeY
		self.reward_range = (0, 1)
		self.rewardStd=rewardStd

		self.nA = 4
		self.nS_all = sizeX * sizeY
		self.nameActions= ["Up", "Down", "Left", "Right"] # Most likely incoherent with the following parts...

		self.initializedRender=False

		#stochastic transitions
		slip=min(slippery,1./3.)
		self.massmap = [[slip, 1.-3*slip, slip, 0., slip],  # up : left up right down stay //old: up down left right stay
				   [slip, 0., slip, 1.-3*slip, slip],  # down
				   [1.-3*slip, slip, 0., slip, slip],  # left
				   [0., slip, 1.-3*slip, slip, slip]]  # right

		if (map_name=="2-room_4corner"):
			self.maze=twoRoom(sizeX, sizeY)
		elif(map_name=="2-room_3corner"):
			self.maze=twoRoom(sizeX, sizeY)
		elif(map_name=="2-room_2corner"):
			self.maze=twoRoom(sizeX, sizeY)
		elif(map_name=="2-room_1corner"):
			self.maze=twoRoom(sizeX, sizeY)
		else:
			print("Name Error, using 2-room_1corner by default")
			self.maze=twoRoom(sizeX, sizeY)

		self.mapping = []
		for x in range(sizeX):
			for y in range(sizeY):
				if self.maze[x, y] >= 1:
					self.mapping.append(self.to_s((x, y)))
		
		self.nS = len(self.mapping)

		#self.goalstates=self.makeGoalStates(nbGoals)
		if (initialSingleStateDistribution):
			isd=self.makeInitialSingleStateDistribution(self.maze)
		else:
			isd=self.makeInitialDistribution(self.maze)
		P = self.makeTransition(isd)
		#R = self.makeRewards()

		self.P = P
		#self.R = R
		self.isd = isd
		self.lastaction=None # for rendering

		self.states = range(0,self.nS)
		self.actions = range(0,self.nA)
		self.nameActions = list(string.ascii_uppercase)[0:min(self.nA,26)]


		#self.reward_range = (0, 1)
		self.action_space = spaces.Discrete(self.nA)
		self.observation_space = spaces.Discrete(self.nS)

		self.initializedRender=False
		self.seed()

		if (map_name=="2-room_4corner"):
			e, t, r = RM_twoRoom_4corner(sizeX, sizeY)
			self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)
		elif(map_name=="2-room_3corner"):
			e, t, r = RM_twoRoom_3corner(sizeX, sizeY)
			self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)
		elif(map_name=="2-room_2corner"):
			e, t, r = RM_twoRoom_2corner(sizeX, sizeY)
			self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)
		elif(map_name=="2-room_1corner"):
			e, t, r = RM_twoRoom_1corner(sizeX, sizeY)
			self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)
		else:
			print("Name Error, using 2-room_1corner by default")
			e, t, r = RM_twoRoom_1corner(sizeX, sizeY)
			self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)

		self.reset()


	def step(self, a):
		old_s = self.s
		transitions = self.P[self.s][a]
		i = categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, d= transitions[i]
		self.s = s
		self.lastaction=a
		s = self.mapping.index(s)
		old_s = self.mapping.index(old_s)
		event = self.rewardMachine.events[old_s, a]
		r = self.rewardMachine.next_step(event)
		return (s, r, d, "")

	def reset(self):
		self.rewardMachine.reset()
		self.s = categorical_sample(self.isd, self.np_random)
		self.lastaction=None
		return self.mapping.index(self.s)








class RiverSwim_patrol2_sas(environments.discreteMDP.DiscreteMDP):
	def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
		self.nS = nbStates
		self.nA = 2
		self.states = range(0,nbStates)
		self.actions = range(0,self.nA)
		self.nameActions = ["R", "L"]


		self.startdistribution = np.zeros((self.nS))
		self.startdistribution[0] =1.
		self.rewards = {}
		self.P = {}
		self.transitions = {}
		# Initialize a randomly generated MDP
		for s in self.states:
			self.P[s]={}
			self.transitions[s]={}
			# GOING RIGHT
			self.transitions[s][0]={}
			self.P[s][0]= [] #0=right", 1=left
			li = self.P[s][0]
			prr=0.
			if (s<self.nS-1):
				li.append((rightProbaright, s+1, False))
				self.transitions[s][0][s+1]=rightProbaright
				prr=rightProbaright
			prl = 0.
			if (s>0):
				li.append((rightProbaLeft, s-1, False))
				self.transitions[s][0][s-1]=rightProbaLeft
				prl=rightProbaLeft
			li.append((1.-prr-prl, s, False))
			self.transitions[s][0][s ] = 1.-prr-prl

			# GOING LEFT
			#if ergodic:
			#	pll = 0.95
			#else:
			#	pll = 1
			self.P[s][1] = []  # 0=right", 1=left
			self.transitions[s][1]={}
			li = self.P[s][1]
			if (s > 0):
				li.append((1., s - 1, False))
				self.transitions[s][1][s-1]=1.
			else:
				li.append((1., s, False))
				self.transitions[s][1][s]=1.

			self.rewards[s]={}
			if (s==self.nS-1):
				self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
			else:
				self.rewards[s][0] = stat.norm(loc=0., scale=0.)
			if (s==0):
				self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
			else:
				self.rewards[s][1] = stat.norm(loc=0., scale=0.)
				
		#print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

		e, t, r = RM_riverSwim_patrol2(nbStates)
		self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)

		super(RiverSwim_patrol2_sas, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)

	def reset(self):
		self.rewardMachine.reset()
		self.s = categorical_sample(self.isd, self.np_random)
		self.lastaction=None
		return self.s

	def step(self, a):
		transitions = self.P[self.s][a]
		i = categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, d= transitions[i]
		event = self.rewardMachine.events[self.s, a, s]
		r =  self.rewardMachine.next_step(event)
		self.s = s
		self.lastaction=a
		return (s, r, d, "")











class RiverSwim_patrol2(environments.discreteMDP.DiscreteMDP):
	def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
		self.nS = nbStates
		self.nA = 2
		self.states = range(0,nbStates)
		self.actions = range(0,self.nA)
		self.nameActions = ["R", "L"]


		self.startdistribution = np.zeros((self.nS))
		self.startdistribution[0] =1.
		self.rewards = {}
		self.P = {}
		self.transitions = {}
		# Initialize a randomly generated MDP
		for s in self.states:
			self.P[s]={}
			self.transitions[s]={}
			# GOING RIGHT
			self.transitions[s][0]={}
			self.P[s][0]= [] #0=right", 1=left
			li = self.P[s][0]
			prr=0.
			if (s<self.nS-1) and (s>0):
				li.append((rightProbaright, s+1, False))
				self.transitions[s][0][s+1]=rightProbaright
				prr=rightProbaright
			elif (s==0):													# To have 0.6 on the leftmost state
				li.append((0.6, s+1, False))
				self.transitions[s][0][s+1]=0.6
				prr=0.6
			prl = 0.
			if (s>0) and (s<self.nS-1):									   # MODIFY HERE FOR THE RIGTHMOST 0.95 and leftmost 0.35
				li.append((rightProbaLeft, s-1, False))
				self.transitions[s][0][s-1]=rightProbaLeft
				prl=rightProbaLeft
			elif s==self.nS-1:												  # To have 0.6 and 0.4 on rightmost state
				li.append((0.4, s-1, False))
				self.transitions[s][0][s-1]=0.4
				prl=0.4
			li.append((1.-prr-prl, s, False))
			self.transitions[s][0][s ] = 1.-prr-prl

			# GOING LEFT
			#if ergodic:
			#	pll = 0.95
			#else:
			#	pll = 1
			self.P[s][1] = []  # 0=right", 1=left
			self.transitions[s][1]={}
			li = self.P[s][1]
			if (s > 0):
				li.append((1., s - 1, False))
				self.transitions[s][1][s-1]=1.
			else:
				li.append((1., s, False))
				self.transitions[s][1][s]=1.

			self.rewards[s]={}
			if (s==self.nS-1):
				self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
			else:
				self.rewards[s][0] = stat.norm(loc=0., scale=0.)
			if (s==0):
				self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
			else:
				self.rewards[s][1] = stat.norm(loc=0., scale=0.)
				
		#print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

		e, t, r = RM_riverSwim_patrol2(nbStates)
		self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)

		super(RiverSwim_patrol2, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)

	def reset(self):
		self.rewardMachine.reset()
		self.s = categorical_sample(self.isd, self.np_random)
		self.lastaction=None
		return self.s

	def step(self, a):
		transitions = self.P[self.s][a]
		i = categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, d= transitions[i]
		event = self.rewardMachine.events[s, a]
		r =  self.rewardMachine.next_step(event)
		self.s = s
		self.lastaction=a
		return (s, r, d, "")
		











class RiverSwim_patrol2_s(environments.discreteMDP.DiscreteMDP):
	def __init__(self, nbStates, rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1.):#, ergodic=False): # TODO ergordic option
		self.nS = nbStates
		self.nA = 2
		self.states = range(0,nbStates)
		self.actions = range(0,self.nA)
		self.nameActions = ["R", "L"]


		self.startdistribution = np.zeros((self.nS))
		self.startdistribution[0] =1.
		self.rewards = {}
		self.P = {}
		self.transitions = {}
		# Initialize a randomly generated MDP
		for s in self.states:
			self.P[s]={}
			self.transitions[s]={}
			# GOING RIGHT
			self.transitions[s][0]={}
			self.P[s][0]= [] #0=right", 1=left
			li = self.P[s][0]
			prr=0.
			if (s<self.nS-1) and (s>0):
				li.append((rightProbaright, s+1, False))
				self.transitions[s][0][s+1]=rightProbaright
				prr=rightProbaright
			elif (s==0):													# To have 0.6 on the leftmost state
				li.append((0.6, s+1, False))
				self.transitions[s][0][s+1]=0.6
				prr=0.6
			prl = 0.
			if (s>0) and (s<self.nS-1):									   # MODIFY HERE FOR THE RIGTHMOST 0.95 and leftmost 0.35
				li.append((rightProbaLeft, s-1, False))
				self.transitions[s][0][s-1]=rightProbaLeft
				prl=rightProbaLeft
			elif s==self.nS-1:												  # To have 0.6 and 0.4 on rightmost state
				li.append((0.4, s-1, False))
				self.transitions[s][0][s-1]=0.4
				prl=0.4
			li.append((1.-prr-prl, s, False))
			self.transitions[s][0][s ] = 1.-prr-prl

			# GOING LEFT
			#if ergodic:
			#	pll = 0.95
			#else:
			#	pll = 1
			self.P[s][1] = []  # 0=right", 1=left
			self.transitions[s][1]={}
			li = self.P[s][1]
			if (s > 0):
				li.append((1., s - 1, False))
				self.transitions[s][1][s-1]=1.
			else:
				li.append((1., s, False))
				self.transitions[s][1][s]=1.

			self.rewards[s]={}
			if (s==self.nS-1):
				self.rewards[s][0] = stat.norm(loc=rewardR,scale=0.)
			else:
				self.rewards[s][0] = stat.norm(loc=0., scale=0.)
			if (s==0):
				self.rewards[s][1] = stat.norm(loc=rewardL,scale=0.)
			else:
				self.rewards[s][1] = stat.norm(loc=0., scale=0.)
				
		#print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

		e, t, r = RM_riverSwim_patrol2(nbStates)
		self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)

		super(RiverSwim_patrol2, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)

	def reset(self):
		self.rewardMachine.reset()
		self.s = categorical_sample(self.isd, self.np_random)
		self.lastaction=None
		return self.s

	def step(self, a):
		transitions = self.P[self.s][a]
		i = categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, d= transitions[i]
		event = self.rewardMachine.events[s]
		r =  self.rewardMachine.next_step(event)
		self.s = s
		self.lastaction=a
		return (s, r, d, "")






class Flower(environments.discreteMDP.DiscreteMDP):
	def __init__(self, sizeB, delta):#, ergodic=False): # TODO ergordic option
		self.nS = 6
		self.nA = 2
		self.states = range(0,6)
		self.actions = range(0,2)
		self.nameActions = ["A", "M"]


		self.startdistribution = np.zeros((self.nS))
		self.startdistribution[0] =1.
		self.rewards = {}
		self.P = {}
		self.transitions = {}
		# Initialize a randomly generated MDP
		for s in self.states:
			self.P[s]={}
			self.transitions[s]={}

			# Action A
			self.transitions[s][0]={}
			self.P[s][0]= []
			li = self.P[s][0]
			if (s==0):
				for ss in self.states:
					li.append((1/6, ss, False))
					self.transitions[s][0][ss]=1/6
			else:
				li.append((delta, s, False))
				self.transitions[s][0][s]=delta
				li.append((1 - delta, 0, False))
				self.transitions[s][0][0]=1 - delta

			# Action M
			self.transitions[s][1]={}
			self.P[s][1]= []
			li = self.P[s][1]
			if (s==0):
				for ss in self.states:
					li.append((1/6, ss, False))
					self.transitions[s][0][ss]=1/6
			else:
				li.append((delta, s, False))
				self.transitions[s][0][s]=delta
				li.append((1 - delta, 0, False))
				self.transitions[s][0][0]=1 - delta

		self.rewards[s]={}
				
		#print("Rewards : ", self.rewards, "\nTransitions : ", self.transitions)

		e, t, r = RM_Flower(sizeB)
		self.rewardMachine = environments.rewardMachine.RewardMachine(e, t, r)

		super(Flower, self).__init__(self.nS, self.nA, self.P,  self.rewards, self.startdistribution)

	def reset(self):
		self.rewardMachine.reset()
		self.s = categorical_sample(self.isd, self.np_random)
		self.lastaction=None
		return self.s

	def step(self, a):
		transitions = self.P[self.s][a]
		i = categorical_sample([t[0] for t in transitions], self.np_random)
		p, s, d= transitions[i]
		event = self.rewardMachine.events[s, a]
		r =  self.rewardMachine.next_step(event)
		self.s = s
		self.lastaction=a
		return (s, r, d, "")