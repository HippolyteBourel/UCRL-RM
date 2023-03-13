from learners.UCRL import *
from learners.UCRL2_L import *
import scipy as sp
import numpy as np



class UCRL2_RM(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM = RM
		self.nU = RM.nb_states

	def name(self):
		return "UCRL2-RM"
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate[s, a])
			max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])# Error?
				l += 1
		return max_p
	

	def sorted_indices(self, u0, s_rm, s, a):
		u = np.zeros(self.nS)
		event = self.RM.events[s, a]
		for ss in range(self.nS):
			if event != None:
				u[ss] = u0[self.RM.transitions[s_rm, event], ss]
			else:
				u[ss] = u0[s_rm, ss]
		#if s_rm == 1 and s == 54:
		#	print(u)
		return np.argsort(u)

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# A reward shapping is performed additionnally in order to compasate the scarcity of rewards in MDPRM.
	def EVI(self, p_estimate, epsilon = 0.01, T = 100):
		t = 0
		u0 = np.zeros((self.nU, self.nS))
		u1 = np.zeros((self.nU, self.nS))
		while True:
			stop = True
			t += 1
			for s_rm in range(self.nU):
				for s in range(self.nS):
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, self.sorted_indices(u0, s_rm, s, a), s, a)
						event = self.RM.events[s, a]
						if event != None:
							ns_rm = self.RM.transitions[s_rm, event]
							reward = self.RM.rewards[s_rm, event]
						else:
							ns_rm = s_rm
							reward = 0
						temp =  reward + sum([u * p for (u, p) in zip(u0[ns_rm], max_p)])
						if (a == 0) or (temp > u1[s_rm, s]):
							u1[s_rm, s] = temp
							self.policy[s_rm, s] = a
				#u1[s_rm] = np.array([max(r_shapped[s_rm, s]) for s in range(self.nS)])
				diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
				if (max(diff) - min(diff)) >= epsilon and t < T:
					stop = False
				if t < self.RM.nb_states:
					stop = False
				#print('u:', u1)
			u0 = cp.deepcopy(u1)
			if t == T:
					print("No convergence in the EVI")
			if stop:
				#print("Finishing an EVI at time:", self.t, " with ", t, " steps.")
				#print("r_shapped:", r_shapped)
				#print("u final: ", u0)
				break

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		self.EVI(p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		action = self.policy[RM_state, state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[RM_state, state]
		return action









class UCRL2_RM_Bernstein(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		#self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS, 2))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM = RM
		self.nU = RM.nb_states

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2-RM-B"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]

	def beta(self, n, delta):
		eta = 1.12
		temp = eta * np.log(np.log(max((np.exp(1), n))) * np.log(eta * max(np.exp(1), n)) / (delta * np.log(eta)**2))
		return temp

	def upper_bound(self, n, p_est, bound_max, beta):
		up = p_est + bound_max
		down = p_est
		for _ in range(5):
			m = (up + down)/2
			temp = np.sqrt(2 * beta * m * (1 - m) / n) + beta / (3 * n)
			if m - temp <= p_est:
				down = m
			else:
				up = m
		return (up + down)/2

	
	def lower_bound(self, n, p_est, bound_max, beta):
		down = p_est - bound_max
		up= p_est
		for _ in range(5):
			m = (up + down)/2
			temp = np.sqrt(2 * beta * m * (1 - m) / n) + beta / (3 * n)
			if m + temp >= p_est:
				up = m
			else:
				down = m
		return (up + down)/2
		

	def distances(self, p_estimate):
		delta = self.delta / (2 * self.nS * self.nA)
		for s in range(self.nS):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				for next_s in range(self.nS):
					p = p_estimate[s, a, next_s]
					beta = self.beta(n, delta)
					bound_max = np.sqrt(beta / (2 * n)) + beta / (3 * n)
					lower_bound = self.lower_bound(n, p, bound_max, beta)
					upper_bound = self.upper_bound(n, p, bound_max, beta)
					self.p_distances[s, a, next_s, 0] = lower_bound
					self.p_distances[s, a, next_s, 1] = upper_bound
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		max_p = np.zeros(self.nS)
		delta = 1.
		for next_s in range(self.nS):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > 0) and (l <= self.nS - 1):
			idx = self.nS - 1 - l if not reverse else l
			idx = sorted_indices[idx]
			new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
			max_p[idx] += new_delta
			delta += - new_delta
			l += 1
		return max_p
	

	def sorted_indices(self, u0, s_rm, s, a):
		u = np.zeros(self.nS)
		event = self.RM.events[s, a]
		for ss in range(self.nS):
			if event != None:
				u[ss] = u0[self.RM.transitions[s_rm, event], ss]
			else:
				u[ss] = u0[s_rm, ss]
		#if s_rm == 1 and s == 54:
		#	print(u)
		return np.argsort(u)

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# A reward shapping is performed additionnally in order to compasate the scarcity of rewards in MDPRM.
	def EVI(self, p_estimate, epsilon = 0.01, T = 100):
		t = 0
		u0 = np.zeros((self.nU, self.nS))
		u1 = np.zeros((self.nU, self.nS))
		#print("initialization:",  r_shapped)
		while True:
			stop = True
			t += 1
			for s_rm in range(self.nU):
				for s in range(self.nS):
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, self.sorted_indices(u0, s_rm, s, a), s, a)
						event = self.RM.events[s, a]
						if event != None:
							ns_rm = self.RM.transitions[s_rm, event]
							reward = self.RM.rewards[s_rm, event]
						else:
							ns_rm = s_rm
							reward = 0
						temp =  reward + sum([u * p for (u, p) in zip(u0[ns_rm], max_p)])
						if (a == 0) or (temp > u1[s_rm, s]):
							u1[s_rm, s] = temp
							self.policy[s_rm, s] = a
				#u1[s_rm] = np.array([max(r_shapped[s_rm, s]) for s in range(self.nS)])
				diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
				if (max(diff) - min(diff)) >= epsilon and t < T:
					stop = False
				if t < self.RM.nb_states:
					stop = False
				#print('u:', u1)
			u0 = cp.deepcopy(u1)
			if t == T:
					print("No convergence")
			if stop:
				#print("Finishing an EVI at time:", self.t, " with ", t, " steps.")
				#print("r_shapped:", r_shapped)
				#print("u final: ", u0)
				break

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances(p_estimate)
		self.EVI(p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		#self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA, self.nS, 2))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		action = self.policy[RM_state, state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[RM_state, state]
		return action






















	



class UCRL2_RM_old(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM = RM
		self.nU = RM.nb_states

	def name(self):
		return "UCRL2-RM"
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate[s, a])
			max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])# Error?
				l += 1
		return max_p
	
	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		res = np.zeros((self.nU, self.nS, self.nA))
		for u in range(self.nU):
			for s in range(self.nS):
				for a in range(self.nA):
					rewards = []
					for ss in range(self.nS):
						event = self.RM.events[ss]
						if event != None:
							rewards += [self.RM.rewards[u, event]]
						else:
							rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					res[u, s, a] = sum([p * r for (p, r) in zip(rewards, max_p)])
		return res

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# A reward shapping is performed additionnally in order to compasate the scarcity of rewards in MDPRM.
	def EVI(self, p_estimate, epsilon = 0.01, T = 100):
		t = 0
		u0 = np.zeros((self.nU, self.nS))
		u1 = np.zeros((self.nU, self.nS))
		sorted_indices = [np.arange(self.nS) for _ in range(self.nU)]
		r_shapped = self.init_r_shapped(p_estimate)
		r_init = cp.deepcopy(r_shapped)
		#print("initialization:",  r_shapped)
		while True:
			stop = True
			t += 1
			for s_rm in range(self.nU):
				for s in range(self.nS):
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices[s_rm], s, a)
						temp = r_init[s_rm, s, a]
						for ss in range(self.nS):
							if self.RM.events[ss] != None:
								temp += max_p[ss] * (max(r_shapped[self.RM.transitions[s_rm, self.RM.events[ss]], ss]) + self.RM.rewards[s_rm, self.RM.events[ss]])
							else:
								temp += max_p[ss] * (max(r_shapped[s_rm, ss]))
						r_shapped[s_rm, s, a] = temp
						temp = r_shapped[s_rm, s, a] + sum([u * p for (u, p) in zip(u0[s_rm], max_p)])
						if (a == 0) or (temp > u1[s_rm, s]):
							u1[s_rm, s] = temp
							self.policy[s_rm, s] = a
				diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
				if (max(diff) - min(diff)) >= epsilon and t < T:
					stop = False
				u0[s_rm] = u1[s_rm]
				u1[s_rm] = np.zeros(self.nS)
				sorted_indices[s_rm] = np.argsort(u0[s_rm])
			if t == T:
					print("No convergence")
			if stop:
				print("Finishing an EVI-RS at time:", self.t)
				#print("r_shapped:", r_shapped)
				#print("u: ", u0)
				break

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		self.EVI(p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		action = self.policy[RM_state, state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[RM_state, state]
		return action











class UCRL2_ORS(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM = RM
		self.nU = RM.nb_states

	def name(self):
		return "UCRL2-ORS"
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS)
		if min1 == 1:
			max_p[sorted_indices[-1]] = 1
		else:
			max_p = cp.deepcopy(p_estimate[s, a])
			max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])# Error?
				l += 1
		return max_p
	
	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		#print("Events:", self.RM.events)
		res = np.zeros((self.nU, self.nS, self.nA))
		for u in range(self.nU):
			for s in range(self.nS):
				for a in range(self.nA):
					rewards = []
					for ss in range(self.nS):
						event = self.RM.events[ss]
						if event != None:
							rewards += [self.RM.rewards[u, event]]
						else:
							rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					res[u, s, a] = sum([p * r for (p, r) in zip(rewards, max_p)])
		return res

	def sorted_indices(self, u0, s_rm, s, a):
		u = np.zeros(self.nS)
		for ss in range(self.nS):
			event = self.RM.events[ss]
			if event != None:
				u[ss] = u0[self.RM.transitions[s_rm, event], ss]
			else:
				u[ss] = u0[s_rm, ss]
		#if s_rm == 1 and s == 54:
		#	print(u)
		return np.argsort(u)

	# Optimistic Reward Shapping algorithm inspired from the EVI and rewrad shapping
	def ORS(self, p_estimate, epsilon = 0.01, T = 100):
		t = 0
		u0 = np.zeros((self.nU, self.nS))
		u1 = np.zeros((self.nU, self.nS))
		sorted_indices = [np.arange(self.nS) for _ in range(self.nU)]
		r_shapped = self.init_r_shapped(p_estimate)
		r_init = cp.deepcopy(r_shapped)
		#print("initialization:",  r_shapped)
		while True:
			stop = True
			t += 1
			for s_rm in range(self.nU):
				if t == 1:
					u0[s_rm] = np.array([max(r_shapped[s_rm, s]) for s in range(self.nS)])
					#print("init:", u0[s_rm])
				#sorted_indices[s_rm] = np.argsort(u0[s_rm]) Not true because ignore the transition between RM states
				for s in range(self.nS):
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, self.sorted_indices(u0, s_rm, s, a), s, a)
						temp = r_init[s_rm, s, a]
						for ss in range(self.nS):
							if self.RM.events[ss] != None:
								temp += max_p[ss] * u0[self.RM.transitions[s_rm, self.RM.events[ss]], ss]#(max(r_shapped[self.RM.transitions[s_rm, self.RM.events[s, a, ss]], ss]))# + self.RM.rewards[s_rm, self.RM.events[s, a, ss]])
							else:
								temp += max_p[ss] * u0[s_rm, ss]#(max(r_shapped[s_rm, ss]))
						r_shapped[s_rm, s, a] = temp
						if (a == 0) or (temp > u1[s_rm, s]):
							u1[s_rm, s] = temp
				#u1[s_rm] = np.array([max(r_shapped[s_rm, s]) for s in range(self.nS)])
				diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
				if (max(diff) - min(diff)) >= epsilon and t < T:
					stop = False
				if t < self.RM.nb_states:
					stop = False
				#print('u:', u1)
			u0 = cp.deepcopy(u1)
			if t == T:
					print("No convergence")
			if stop:
				print("Finishing an ORS at time:", self.t, " with ", t, " steps.")
				#print("r_shapped:", r_shapped)
				#print("u final: ", u0)
				break
		return r_shapped

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# A reward shapping is performed additionnally in order to compasate the scarcity of rewards in MDPRM.
	def EVI(self, p_estimate, r_shapped, epsilon = 0.01, T = 100):
		t = 0
		u0 = np.zeros((self.nU, self.nS))
		u1 = np.zeros((self.nU, self.nS))
		sorted_indices = [np.arange(self.nS) for _ in range(self.nU)]
		#print("initialization:",  r_shapped)
		while True:
			stop = True
			t += 1
			for s_rm in range(self.nU):
				for s in range(self.nS):
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices[s_rm], s, a)
						temp = r_shapped[s_rm, s, a] + sum([u * p for (u, p) in zip(u0[s_rm], max_p)])
						if (a == 0) or (temp > u1[s_rm, s]):
							u1[s_rm, s] = temp
							self.policy[s_rm, s] = a
				diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
				if (max(diff) - min(diff)) >= epsilon and t < T:
					stop = False
				u0[s_rm] = u1[s_rm]
				u1[s_rm] = np.zeros(self.nS)
				sorted_indices[s_rm] = np.argsort(u0[s_rm])
			if t == T:
					print("No convergence")
			if stop:
				print("Finishing an EVI at time:", self.t)
				#print("r_shapped:", r_shapped)
				#print("u: ", u0)
				break

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nS, self.nA))
		p_estimate = np.zeros((self.nS, self.nA, self.nS))
		for s in range(self.nS):
			for a in range(self.nA):
				div = max([1, self.Nk[s, a]])
				for next_s in range(self.nS):
					p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		r_shapped = self.ORS(p_estimate)
		self.EVI(p_estimate, r_shapped)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.policy = np.zeros((self.nU, self.nS,), dtype=int)
		self.r_distances = np.zeros((self.nS, self.nA))
		self.p_distances = np.zeros((self.nS, self.nA))
		self.Pk = np.zeros((self.nS, self.nA, self.nS))
		self.Rk = np.zeros((self.nS, self.nA))
		self.u = np.zeros(self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		action = self.policy[RM_state, state]
		if self.vk[state, action] >= max([1, self.Nk[state, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[RM_state, state]
		return action


