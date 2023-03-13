from environments import discreteMDP, MDPRM_library
import pylab as pl
import gym
import pickle
from gym.envs.registration import  register
import numpy as np
from learners.UCRL_RM import *
from learners.UCRL_CP import *
from learners.Optimal import *
from environments import equivalence
#from learners.ImprovedMDPLearner2 import *
from utils import *

# Auxialiary for SA-MDP
def make_cp_transitions(RM, P, nS, nA):
	nQ = RM.nb_states
	res = np.zeros((nQ, nS, nA, nQ, nS))
	for q in range(nQ):
		for s in range(nS):
			for a in range(nA):
				for qq in range(nQ):
					for ss in range(nS):
						event = RM.events[s, a]
						if event == None:
							q_transition = q
						else:
							q_transition = RM.transitions[q, event]
						if qq == q_transition:
							res[q, s, a, qq, ss] = P[s, a, ss]
	return res

def make_transitions(P, nS, nA):
	res = np.zeros((nS, nA, nS))
	for s in range(nS):
		for a in range(nA):
			for e in P[s][a]:
				res[s, a, e[1]] = e[0]
	return res

def make_transitions_grid(env, P, nS, nA):
	res = np.zeros((nS, nA, nS))
	for s in range(nS):
		for a in range(nA):
			for e in P[env.mapping[s]][a]:
				ss = env.mapping.index(e[1])
				res[s, a, ss] = e[0]
	return res

def build_Bqs(RM, P, nS, nA):
	nQ = RM.nb_states
	res = np.zeros((nQ, nS, nQ))
	for q in range(nQ):
		for s in range(nS):
			for qq in range(nQ):
				p = max([max(P[q, s, a, qq]) for a in range(nA)])
				if p > 0:
					res[q, s, qq] = 1
	return res
			

def build_Ksa(P, nS, nA):
	Ksa = np.zeros((nS, nA, nS))
	for s in range(nS):
		for a in range(nA):
			for ss in range(nS):
				if P[s, a, ss] > 0:
					Ksa[s, a, ss] = 1
	return Ksa

def build_Kqsa(RM, P, nS, nA):
	nQ = RM.nb_states
	Kqsa = np.zeros((nQ, nS, nA, nQ, nS))
	for q in range(nQ):
		for s in range(nS):
			for a in range(nA):
				for qq in range(nQ):
					for ss in range(nS):
						if P[q, s, a, qq, ss] > 0:
							Kqsa[q, s, a, qq, ss] = 1
	return Kqsa


# Return diameter of an MDP
def diameter_mdp(env, epsilon = 0.01, max_iter = 1000):
	D = 0
	u0 = np.zeros(env.nS)
	u1 = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	for sD in range(env.nS):
		niter = 0
		while True:
			niter += 1
			for s in range(env.nS):
				for a in range(env.nA):
					r = 0
					if s == sD:
						r = 1
						temp = r + u0[s]
					else:
						temp = r + sum([u * p for (u, p) in zip(u0, P[s, a])])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				D = max(D, max(u1) - min(u1))
				break
			else:
				u0 = u1
				u1 = np.zeros(env.nS)
			if niter > max_iter:
				D = max(D, max(u0) - min(u0))
				print("No convergence in VI")
				break
	return D


def aux_local_mdp(env, epsilon, P, Ksa, max_iter = 1000):
	D = 0
	u0 = np.zeros(env.nS)
	u1 = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	s_list = []
	for ss in range(env.nS):
		for a in range(env.nA):
			if Ksa[a, ss] == 1:
				s_list.append(ss)
				break
	for sD in s_list:
		niter = 0
		while True:
			niter += 1
			for s in range(env.nS):
				for a in range(env.nA):
					r = 0
					if s == sD:
						r = 1
						temp = r + u0[s]
					else:
						temp = r + sum([u * p for (u, p) in zip(u0, P[s, a])])
					if (a == 0) or (temp > u1[s]):
						u1[s] = temp
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				temp = [u1[s] for s in s_list]  
				D = max(D, np.max(temp) - np.min(temp))
				break
			else:
				u0 = u1
				u1 = np.zeros(env.nS)
			if niter > max_iter:
				temp = [u0[s] for s in s_list]  
				D = max(D, np.max(temp) - np.min(temp))
				print("No convergence in VI")
				break
	return D


# Return 'local' diameterS for an MDP
def local_diameter_mdp(env, epsilon = 0.01):
	D = np.zeros(env.nS)
	P = make_transitions(env.P, env.nS, env.nA)
	Ksa = build_Ksa(P, env.nS, env.nA)
	for S in range(env.nS):
		D[S] = aux_local_mdp(env, epsilon, P, Ksa[S])
	return D

# Return diameter of an SA-MDP
def diameter_samdp(env, grid = False, epsilon = 0.01, max_iter = 10000):
	D = 0
	RM = env.rewardMachine
	nQ = RM.nb_states
	u0 = np.zeros((nQ, env.nS))
	u1 = np.zeros((nQ, env.nS))
	if grid:
		temp_P = make_transitions_grid(env, env.P, env.nS, env.nA)
	else:
		temp_P = make_transitions(env.P, env.nS, env.nA)
	P = make_cp_transitions(RM, temp_P, env.nS, env.nA)
	for qD in range(nQ):
		for sD in range(env.nS):
			niter = 0
			while True:
				niter += 1
				for q in range(nQ):
					for s in range(env.nS):
						for a in range(env.nA):
							r = 0
							if q == qD and s == sD:
								r = 1
								temp = 1 + u0[q, s]
							else:
								somme = 0
								for qq in range(nQ):
									for ss in range(env.nS):
										somme += u0[qq, ss] * P[q, s, a, qq, ss]
								temp = r + somme
							if (a == 0) or (temp > u1[q, s]):
								u1[q, s] = temp
				diff = [[] for _ in range(nQ)]
				for q in range(nQ):
					diff[q]  = [abs(x - y) for (x, y) in zip(u1[q], u0[q])]
				if (np.max(diff) - np.min(diff)) < epsilon:
					D = max(D, np.max(u1) - np.min(u1))
					break
				else:
					#print(np.max(u1), np.min(u1), np.max(u1) - np.min(u1))
					#print(u1)
					u0 = u1
					u1 = np.zeros((nQ, env.nS))
				if niter > max_iter:
					D = max(D, np.max(u0) - np.min(u0))
					print("No convergence in VI")
					break
	return D




def aux_local_samdp(env, epsilon, P, nQ, Bqs, max_iter = 1000):
	D = 0
	u0 = np.zeros((nQ, env.nS))
	u1 = np.zeros((nQ, env.nS))
	q_list = []
	for q in range(nQ):
		if Bqs[q] == 1:
			q_list.append(q)
	for qD in q_list:
		for sD in range(env.nS):
			niter = 0
			while True:
				niter += 1
				for q in range(nQ):
					for s in range(env.nS):
						for a in range(env.nA):
							r = 0
							if q == qD and s == sD:
								r = 1
								temp = 1 + u0[q, s]
							else:
								somme = 0
								for qq in range(nQ):
									for ss in range(env.nS):
										somme += u0[qq, ss] * P[q, s, a, qq, ss]
								temp = r + somme
							if (a == 0) or (temp > u1[q, s]):
								u1[q, s] = temp
				diff = [[] for _ in range(nQ)]
				for q in range(nQ):
					diff[q]  = [abs(x - y) for (x, y) in zip(u1[q], u0[q])]
				if (np.max(diff) - np.min(diff)) < epsilon:
					temp = [u1[q] for q in q_list]  
					D = max(D, np.max(temp) - np.min(temp))
					break
				else:
					u0 = u1
					u1 = np.zeros((nQ, env.nS))
				if niter > max_iter:
					temp = [u0[q] for q in q_list]  
					D = np.max(D, np.max(temp) - np.min(temp))
					print("No convergence in VI")
					break
	return D

# Return 'new' diameterS for an SA-MDP
def local_diameter_samdp(env, grid = False, epsilon = 0.01):
	D = np.zeros(env.nS)
	RM = env.rewardMachine
	nQ = RM.nb_states
	if grid:
		temp_P = make_transitions_grid(env, env.P, env.nS, env.nA)
	else:
		temp_P = make_transitions(env.P, env.nS, env.nA)
	P = make_cp_transitions(RM, temp_P, env.nS, env.nA)
	Bqs = build_Bqs(RM, P, env.nS, env.nA)
	for S in range(env.nS):
		for Q in range(nQ):
			tempD = aux_local_samdp(env, epsilon, P, nQ, Bqs[Q, S])
			D[S] = max(D[S], tempD)
	return D


# Local diameter of SA-MDP
def aux_local_bernstein_samdp(env, epsilon, P, nQ, Kqsa, max_iter = 1000):
	D = 0
	u0 = np.zeros((nQ, env.nS))
	u1 = np.zeros((nQ, env.nS))
	qs_list = []
	for q in range(nQ):
		for s in range(env.nS):
			for a in range(env.nA):
				if Kqsa[a, q, s] == 1:
					qs_list.append([q, s])
					break
	for [qD, sD] in qs_list:
			niter = 0
			while True:
				niter += 1
				for q in range(nQ):
					for s in range(env.nS):
						for a in range(env.nA):
							r = 0
							if q == qD and s == sD:
								r = 1
								temp = 1 + u0[q, s]
							else:
								somme = 0
								for qq in range(nQ):
									for ss in range(env.nS):
										somme += u0[qq, ss] * P[q, s, a, qq, ss]
								temp = r + somme
							if (a == 0) or (temp > u1[q, s]):
								u1[q, s] = temp
				diff = [[] for _ in range(nQ)]
				for q in range(nQ):
					diff[q]  = [abs(x - y) for (x, y) in zip(u1[q], u0[q])]
				if (np.max(diff) - np.min(diff)) < epsilon:
					temp = [u1[q, s] for [q, s] in qs_list]  
					D = max(D, np.max(temp) - np.min(temp))
					break
				else:
					u0 = u1
					u1 = np.zeros((nQ, env.nS))
				if niter > max_iter:
					temp = [u0[q, s] for [q, s] in qs_list]  
					D = max(D, np.max(temp) - np.min(temp))
					print("No convergence in VI")
					break
	return D

# Return local diameterS for an SA-MDP
def local_berstein_diameter_samdp(env, grid = False, epsilon = 0.01):
	D = np.zeros((env.rewardMachine.nb_states, env.nS))
	Ds = np.zeros(env.nS)
	RM = env.rewardMachine
	nQ = RM.nb_states
	if grid:
		temp_P = make_transitions_grid(env, env.P, env.nS, env.nA)
	else:
		temp_P = make_transitions(env.P, env.nS, env.nA)
	P = make_cp_transitions(RM, temp_P, env.nS, env.nA)
	Kqsa = build_Kqsa(RM, P, env.nS, env.nA)
	for S in range(env.nS):
		for Q in range(nQ):
			tempD = aux_local_bernstein_samdp(env, epsilon, P, nQ, Kqsa[Q, S])
			D[Q, S] = tempD
			Ds[S] = max(Ds[S], tempD)
	return D, Ds


def run_exp_samdp(testName = "riverSwim6_patrol2", S=6, delta = 0.2):

	if testName == "2-room_patrol2":
		env, nbS, nbA = buildGridworld_RM(sizeX=9,sizeY=11,map_name="2-room_patrol2",rewardStd=0.01, initialSingleStateDistribution=True)
	elif testName == "riverSwim6_patrol2":
		env, nbS, nbA = buildRiverSwim_patrol2(nbStates=S, rightProbaright=0.35, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "riverSwim25_patrol2":
		env, nbS, nbA = buildRiverSwim_patrol2(nbStates=25, rightProbaright=0.35, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName ==  "Flower":
		env, nbS, nbA = buildFlower(sizeB = S, delta = delta)

	D, Ds = diameter_samdp(env, grid = False), local_diameter_samdp(env, grid = False)
	Dqs, Ds_b = local_berstein_diameter_samdp(env, grid = False)
	print("Cross-product diameter : Dcp = ", D)
	print("sqrt(S)Dcp = ", np.sqrt(nbS)*D)
	print("New diameter : Ds = ", Ds)
	print("sum Ds^2 = ", sum([d**2 for d in Ds]))
	print("Local diameter : Dqs = ", Ds_b)
	print("sum Ds^2 = ", sum([d**2 for d in Ds_b]))
	#print("sum Ds^2 = ", sum([d**2 for d in Ds]))
	return D, Ds



def run_exp_samdp_grid(testName = "riverSwim6_patrol2", S=6):
	X = 5
	Y = 9
	if testName == "2-room_1corner":
		env, nbS, nbA = buildGridworld_RM(X, Y, "2-room_1corner")
	elif testName == "2-room_2corner":
		env, nbS, nbA = buildGridworld_RM(X, Y, "2-room_2corner")
	elif testName == "2-room_3corner":
		env, nbS, nbA = buildGridworld_RM(X, Y, "2-room_3corner")
	elif testName == "2-room_4corner":
		env, nbS, nbA = buildGridworld_RM(X, Y, "2-room_4corner")

	D, Ds = diameter_samdp(env, grid = True), local_diameter_samdp(env, True)
	Dqs, Ds_b = local_berstein_diameter_samdp(env, grid = True)
	print("Cross-product diameter : Dcp = ", D)
	print("sqrt(S)Dcp = ", np.sqrt(7)*D)# 7 states in the flower env.
	print("New diameter : Ds = ", Ds)
	print("sum Ds^2 = ", sum([d**2 for d in Ds]))
	print("Local diameter : Dqs = ", Ds_b)
	print("sum Ds^2 = ", sum([d**2 for d in Ds_b]))
	#print("sum Ds^2 = ", sum([d**2 for d in Ds]))
	return D, Ds



def run_exp_mdp(testName = "riverSwim"):
	
	if testName == "random_grid":
		env, nbS, nbA = buildGridworld(sizeX=10,sizeY=10,map_name="random",rewardStd=0.01, initialSingleStateDistribution=True)
	elif testName == "2-room":
		env, nbS, nbA = buildGridworld(sizeX=9, sizeY=11, map_name="2-room", rewardStd=0.0, initialSingleStateDistribution=True)
	elif testName == "4-room":
		env, nbS, nbA = buildGridworld(sizeX=7, sizeY=7, map_name="4-room", rewardStd=0.0, initialSingleStateDistribution=True)
	elif testName == "random":
		env, nbS, nbA = buildRandomMDP(nbStates=6,nbActions=3, maxProportionSupportTransition=0.25, maxProportionSupportReward=0.1, maxProportionSupportStart=0.2, minNonZeroProbability=0.15, minNonZeroReward=0.3, rewardStd=0.5)
	elif testName == "three-state":
		ns_river = 1
		env, nbS, nbA = buildThreeState(delta = 0.005)
	elif testName == "three-state-bernoulli":
		ns_river = 1
		env, nbS, nbA = buildThreeState(delta = 0.00, fixed_reward = False)
	elif testName == "riverSwimErgo":
		ns_river = 6
		env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "riverSwimErgo25":
		ns_river = 25
		env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "riverSwimErgo50":
		ns_river = 50
		env, nbS, nbA = buildRiverSwimErgo(nbStates=ns_river, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "riverSwim25_shuffle":
		ns_river = 25
		env, nbS, nbA = buildRiverSwim_shuffle(nbStates=ns_river, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.4, rightProbaLeft2=0.05)
	elif testName == "riverSwim25_biClass":
		ns_river = 25
		env, nbS, nbA = buildRiverSwim_biClass(nbStates=ns_river, max_steps=np.infty,reward_threshold=np.infty,rightProbaright=0.6, rightProbaLeft=0.05, rewardL=0.1, rewardR=1., rightProbaright2=0.4, rightProbaLeft2=0.05)
	else:
		if testName == "riverSwim25":
			ns_river = 25
		else:
			ns_river = 100
		env, nbS, nbA = buildRiverSwim(nbStates=ns_river, rightProbaright=0.35, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	
	D = diameter_mdp(env)
	print("Diameter : D = ", D)
	Ds = local_diameter_mdp(env)
	temp = [d**2 for d in Ds]
	print("Local diameter : min(Ds) = ", min(Ds), ", max(Ds) = ", max(Ds), ", sum(Ds^2) = ", sum(temp))
	return D


#print("Vanilla riverSwim 100 states:")
#D = run_exp_mdp()
#for S in [6, 12, 20, 40, 70, 100]:
S = 12
print("Flower with big loop ", S, "-states:")
D, Ds = run_exp_samdp("Flower", S = S)#("riverSwim6_patrol2", S)