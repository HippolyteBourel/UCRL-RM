from learners.UCRL import *
from learners.UCRL2_L import *
import scipy as sp
import numpy as np


class UCRL2_CP(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2(CP)"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]

	def distances(self):
		d = self.delta / (2 * self.nS * self.nU * self.nA)
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					n = max(1, self.Nk[s, a])
					self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nU * self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS * self.nU)
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

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = np.zeros(self.nS * self.nU)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						event = self.RM.events[state, a]
						if event != None:
							reward = self.RM.rewards[s_rm, event]
						else:
							reward = 0
						temp = reward + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))


	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		self.EVI(p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1




class UCRL2_CP_RMVI(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def rv_mapping(self, s):
		u = s//self.nS
		res_s = s%self.nS
		return u, res_s

	def name(self):
		return "UCRL2-CP-RMVI"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]
	
	def updateP(self):
		s = self.observations[0][-2]
		a = self.observations[1][-1]
		ss = self.observations[0][-1]
		self.Pk[s, a, ss] += 1

	def distances(self):
		d = self.delta / (2 * self.nS * self.nU * self.nA)
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					n = max(1, self.Nk[s, a])
					self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS * self.nU)
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

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		r_shapped = self.init_r_shapped(p_estimate)#self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u0 = cp.deepcopy(r_shapped)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						temp = r_shapped[s] + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))

	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		#print("Events:", self.RM.events)
		res = np.zeros(self.nU * self.nS)
		for u in range(self.nU):
			for s in range(self.nS):
				temp = np.zeros(self.nA)
				for a in range(self.nA):
					rewards = []
					for s_rm in range(self.nU):
						for ss in range(self.nS):
							event = self.RM.events[ss]
							if event != None:
								rewards += [self.RM.rewards[u, event]]
							else:
								rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp[a]  = sum([p * r for (p, r) in zip(rewards, max_p)])
				res[self.mapping(u, s)] = max(temp) 
		return res

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		r_estimate = self.init_r_shapped(p_estimate)
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1






class UCRL2_CP_maxp(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2-CP-maxp"

	def rv_mapping(self, s):
		u = s//self.nS
		res_s = s%self.nS
		return u, res_s

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]

	def distances(self):
		d = self.delta / (2 * self.nS * self.nU * self.nA)
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					n = max(1, self.Nk[s, a])
					self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nU * self.nS) - 2) / d)) / n)
	

	def max_proba2(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS * self.nU)
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

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS * self.nU)
		l = -1
		if min1 == 1:
			while min1 == 1:
				candidate_u, ss = self.rv_mapping(sorted_indices[l])
				u, _ = self.rv_mapping(s)
				event = self.RM.events[ss]
				if event != None:
					next_u = self.RM.transitions[u, event]
				else:
					next_u = u
				if next_u == candidate_u:
					max_p[sorted_indices[l]] = 1
					min1 = 0
				else:
					l -= 1
					min1 = min([1, p_estimate[s, a, sorted_indices[l]] + (self.p_distances[s, a] / 2)])
		elif min1 != 0:
			max_p = cp.deepcopy(p_estimate[s, a])
			#l = -1
			#candidate_u, ss = self.rv_mapping(sorted_indices[l])
			#u, _ = self.rv_mapping(s)
			#event = self.RM.events[ss]
			#if event != None:
			#	next_u = self.RM.transitions[u, event]
			#else:
			#	next_u = u
			#while next_u != candidate_u:
			#	l -= 1
			#	candidate_u, ss = self.rv_mapping(sorted_indices[l])
			#	event = self.RM.events[ss]
			#	if event != None:
			#		next_u = self.RM.transitions[u, event]
			#	else:
			#		next_u = u
			max_p[sorted_indices[l]] += self.p_distances[s, a] / 2
			l = 0
			while sum(max_p) > 1:
				max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])
				l += 1
		return max_p

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u1 = np.zeros(self.nS * self.nU)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						temp = self.RM.rewards[s_rm, self.RM.events[state, a]] + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))


	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		r_estimate = self.init_r_shapped(p_estimate)
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1


















class UCRL2_CP_Bernstein(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS, 2))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2(CP)-Bernstein"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
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
		delta = self.delta / (2 * self.nS * self.nA * self.nU)
		for s in range(self.nS * self.nU):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				for next_s in range(self.nS * self.nU):
					p = p_estimate[s, a, next_s]
					beta = self.beta(n, delta)
					bound_max = np.sqrt(beta / (2 * n)) + beta / (3 * n)
					lower_bound = self.lower_bound(n, p, bound_max, beta)
					upper_bound = self.upper_bound(n, p, bound_max, beta)
					self.p_distances[s, a, next_s, 0] = lower_bound
					self.p_distances[s, a, next_s, 1] = upper_bound
					

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		max_p = np.zeros(self.nS * self.nU)
		delta = 1.
		for next_s in range(self.nS * self.nU):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > 0) and (l <= self.nS * self.nU - 1):
			idx = self.nS * self.nU - 1 - l if not reverse else l
			idx = sorted_indices[idx]
			new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
			max_p[idx] += new_delta
			delta += - new_delta
			l += 1
		return max_p

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = np.zeros(self.nS * self.nU)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						event = self.RM.events[state, a]
						if event != None:
							reward = self.RM.rewards[s_rm, event]
						else:
							reward = 0
						temp = reward + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))


	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances(p_estimate)
		self.EVI(p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS, 2))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1









class UCRL2_CP_old(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2-CP"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]

	def distances(self):
		d = self.delta / (2 * self.nS * self.nU * self.nA)
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					n = max(1, self.Nk[s, a])
					self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nU * self.nS) - 2) / d)) / n)
	
	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a):
		min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
		max_p = np.zeros(self.nS * self.nU)
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

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		r_shapped = self.init_r_shapped(p_estimate)#self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u0 = cp.deepcopy(r_shapped)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						temp = r_shapped[s] + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))

	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		#print("Events:", self.RM.events)
		res = np.zeros(self.nU * self.nS)
		for u in range(self.nU):
			for s in range(self.nS):
				temp = np.zeros(self.nA)
				for a in range(self.nA):
					rewards = []
					for s_rm in range(self.nU):
						for ss in range(self.nS):
							event = self.RM.events[ss]
							if event != None:
								rewards += [self.RM.rewards[u, event]]
							else:
								rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp[a]  = sum([p * r for (p, r) in zip(rewards, max_p)])
				res[self.mapping(u, s)] = max(temp) 
		return res

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		r_estimate = self.init_r_shapped(p_estimate)
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1


















class UCRL2_CP_Bernstein_old(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS, 2))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2-CP-Bernstein"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
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
		delta = self.delta / (2 * self.nS * self.nA * self.nU)
		for s in range(self.nS * self.nU):
			for a in range(self.nA):
				n = max(1, self.Nk[s, a])
				for next_s in range(self.nS * self.nU):
					p = p_estimate[s, a, next_s]
					beta = self.beta(n, delta)
					bound_max = np.sqrt(beta / (2 * n)) + beta / (3 * n)
					lower_bound = self.lower_bound(n, p, bound_max, beta)
					upper_bound = self.upper_bound(n, p, bound_max, beta)
					self.p_distances[s, a, next_s, 0] = lower_bound
					self.p_distances[s, a, next_s, 1] = upper_bound
					

	# Computing the maximum proba in the Extended Value Iteration for given state s and action a.
	def max_proba(self, p_estimate, sorted_indices, s, a, epsilon = 10**(-8), reverse = False):
		max_p = np.zeros(self.nS * self.nU)
		delta = 1.
		for next_s in range(self.nS * self.nU):
			max_p[next_s] = max((0, p_estimate[s, a, next_s] - self.p_distances[s, a, next_s, 0]))
			delta += - max_p[next_s]
		l = 0
		while (delta > 0) and (l <= self.nS * self.nU - 1):
			idx = self.nS * self.nU - 1 - l if not reverse else l
			idx = sorted_indices[idx]
			new_delta = min((delta, p_estimate[s, a, idx] + self.p_distances[s, a, idx, 1] - max_p[idx]))
			max_p[idx] += new_delta
			delta += - new_delta
			l += 1
		return max_p

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		r_shapped = self.init_r_shapped(p_estimate)#self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u0 = cp.deepcopy(r_shapped)
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.argsort(u0)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						temp = r_shapped[s] + sum([u * p for (u, p) in zip(u0, max_p)])#r_estimate[s_rm, state, a] + 
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		#print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))

	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		#print("Events:", self.RM.events)
		res = np.zeros(self.nU * self.nS)
		for u in range(self.nU):
			for s in range(self.nS):
				temp = np.zeros(self.nA)
				for a in range(self.nA):
					rewards = []
					for s_rm in range(self.nU):
						for ss in range(self.nS):
							event = self.RM.events[ss]
							if event != None:
								rewards += [self.RM.rewards[u, event]]
							else:
								rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					temp[a]  = sum([p * r for (p, r) in zip(rewards, max_p)])
				res[self.mapping(u, s)] = max(temp) 
		return res

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s_rm in range(self.nU):
						for next_state in range(self.nS):
							next_s = self.mapping(next_s_rm, next_state)
							p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances(p_estimate)
		r_estimate = self.init_r_shapped(p_estimate)
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA, self.nS * self.nU, 2))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1











class UCRL2_CP_sas(UCRL2_L_boost):
	def __init__(self,nS, nA, RM, delta):
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM = RM

	def mapping(self, u, s):
		return u * self.nS + s

	def name(self):
		return "UCRL2-CP"

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		for u in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(u, state)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]

	def distances(self):
		d = self.delta / (2 * self.nS * self.nU * self.nA)
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					n = max(1, self.Nk[s, a])
					self.p_distances[s, a] = np.sqrt((2 * (1 + 1 / n) * np.log(np.sqrt(n + 1) * (2**(self.nU * self.nS) - 2) / d)) / n)
	
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

	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def EVI(self, r_estimate, p_estimate, epsilon = 0.01, max_iter = 1000):
		action_noise = [(np.random.random_sample() * 0.1 * min((1e-6, epsilon))) for _ in range(self.nA)]
		u0 = self.u#np.zeros(self.nS)   #sligthly boost the computation and doesn't seems to change the results
		u1 = np.zeros(self.nS * self.nU)
		sorted_indices = np.arange(self.nS)
		niter = 0
		while True:
			niter += 1
			for s_rm in range(self.nU):
				for state in range(self.nS):
					s = self.mapping(s_rm, state)
					for a in range(self.nA):
						max_p = self.max_proba(p_estimate, sorted_indices, s, a)
						temp = r_estimate[s_rm, state, a] + sum([u * p for (u, p) in zip(u0, max_p)])
						if (a == 0) or ((temp + action_noise[a]) > (u1[s] + action_noise[self.policy[s]])):#(temp > u1[s]):
							u1[s] = temp
							self.policy[s] = a
			diff  = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				break
			else:
				u0 = u1
				u1 = np.zeros(self.nU * self.nS)
				sorted_indices = np.argsort(u0)
			if niter > max_iter:
				print("No convergence in EVI")
				break
		self.u = u0 
		print("Finish EVI at time:", self.t)
		#self.span.append(max(u0) - min(u0))

	# Function to initialise the r-shapped (meaning that we get the real reward)
	def init_r_shapped(self, p_estimate):
		#print("Events:", self.RM.events)
		res = np.zeros((self.nU, self.nS, self.nA))
		for u in range(self.nU):
			for s in range(self.nS):
				for a in range(self.nA):
					rewards = []
					for ss in range(self.nS):
						event = self.RM.events[s, a, ss]
						if event != None:
							rewards += [self.RM.rewards[u, event]]
						else:
							rewards += [0]
					sorted_indices = np.argsort(np.array(rewards))
					max_p = self.max_proba(p_estimate, sorted_indices, s, a)
					res[u, s, a] = sum([p * r for (p, r) in zip(rewards, max_p)])
		return res

	# To start a new episode (init var, computes estmates and run EVI).
	def new_episode(self):
		self.updateN() # Don't run it after the reinitialization of self.vk
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		p_estimate = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		for s_rm in range(self.nU):
			for state in range(self.nS):
				s = self.mapping(s_rm, state)
				for a in range(self.nA):
					div = max([1, self.Nk[s, a]])
					for next_s in range(self.nS):
						p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
		self.distances()
		r_estimate = self.init_r_shapped(p_estimate)
		self.EVI(r_estimate, p_estimate)

	# To reinitialize the learner with a given initial state inistate.
	def reset(self,inistate):
		self.t = 1
		self.tk = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.policy = np.zeros((self.nU * self.nS,), dtype=int)
		self.p_distances = np.zeros((self.nU * self.nS, self.nA))
		self.Pk = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))
		self.u = np.zeros(self.nU * self.nS)
		self.span = []
		self.RM.reset()
		self.new_episode()

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = self.policy[s]
		if self.vk[s, action] >= max([1, self.Nk[s, action]]): # Stoppping criterion
			self.new_episode()
			action  = self.policy[s]
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		p_s_rm = self.RM.previous_state
		s_rm = self.RM.current_state
		s = self.mapping(p_s_rm, state)
		o = self.mapping(s_rm, observation)
		self.vk[s, action] += 1
		self.observations[0].append(o)
		self.observations[1].append(action)
		self.observations[2].append(reward)
		self.updateP()
		self.t += 1







