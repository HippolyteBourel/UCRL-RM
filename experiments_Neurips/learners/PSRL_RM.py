import scipy.stats as stat
import numpy as np

class Agent:
	def __init__(self, nS, nA, name="Agent"):
		self.nS = nS
		self.nA = nA
		self.agentname= name

	def name(self):
		return self.agentname

	def reset(self,inistate):
		()

	def play(self,state):
		return np.random.randint(self.nA)

	def update(self, state, action, reward, observation):
		()

def randamax(V, T=None, I=None):
	"""
	V: array of values
	T: array used to break ties
	I: array of indices from which we should return an amax
	"""
	if I is None:
		idxs = np.where(V == np.amax(V))[0]
		if T is None:
			idx = np.random.choice(idxs)
		else:
			assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
			t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
			t_idxs = np.random.choice(t_idxs)
			idx = idxs[t_idxs]
	else:
		idxs = np.where(V[I] == np.amax(V[I]))[0]
		if T is None:
			idx = I[np.random.choice(idxs)]
		else:
			assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
			t = T[I]
			t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
			t_idxs = np.random.choice(t_idxs)
			idx = I[idxs[t_idxs]]
	return idx


def randamin(V, T=None, I=None):
	"""
	V: array of values
	T: array used to break ties
	I: array of indices from which we should return an amax
	"""
	if I is None:
		idxs = np.where(V == np.amin(V))[0]
		if T is None:
			idx = np.random.choice(idxs)
		else:
			assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
			t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
			t_idxs = np.random.choice(t_idxs)
			idx = idxs[t_idxs]
	else:
		idxs = np.where(V[I] == np.amin(V[I]))[0]
		if T is None:
			idx = I[np.random.choice(idxs)]
		else:
			assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
			t = T[I]
			t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
			t_idxs = np.random.choice(t_idxs)
			idx = I[idxs[t_idxs]]
	return idx



def allmax(a):
	if len(a) == 0:
		return []
	all_ = [0]
	max_ = a[0]
	for i in range(1, len(a)):
		if a[i] > max_:
			all_ = [i]
			max_ = a[i]
		elif a[i] == max_:
			all_.append(i)
	return (max_, all_)


def allmin(a):
	if len(a) == 0:
		return []
	all_ = [0]
	min_ = a[0]
	for i in range(1, len(a)):
		if a[i] < min_:
			all_ = [i]
			min_ = a[i]
		elif a[i] == min_:
			all_.append(i)
	return (min_, all_)


def categorical_sample(prob_n, np_random):
	"""
	Sample from categorical distribution
	Each row specifies class probabilities
	"""
	prob_n = np.asarray(prob_n)
	csprob_n = np.cumsum(prob_n)
	return (csprob_n > np_random.rand()).argmax()


def kl(x, y):
	if (x == 0):
		if (y == 1.):
			return np.infty
		return np.log(1. / (1. - y))
	if (x == 1):
		if (y == 0.):
			return np.infty
		return np.log(1. / y)
	if (y == 0) or (y == 1):
		return np.infty
	return x * np.log(x / y) + (1. - x) * np.log((1. - x) / (1. - y))


def search_up(f, up, down, epsilon=0.0001):
	mid = (up + down) / 2
	if (up - down > epsilon):
		if f(mid):
			return search_up(f, up, mid)
		else:
			return search_up(f, mid, down)
	else:
		if f(up):
			return up
		return down


def search_down(f, up, down, epsilon=0.0001):
	mid = (up + down) / 2
	if (up - down > epsilon):
		if f(mid):
			return search_down(f, mid, down)
		else:
			return search_down(f, up, mid)
	else:
		if f(down):
			return down
		return up




class TSDE_CP(Agent):
	def __init__(self, nS, nA, RM, delta):
		Agent.__init__(self, nS, nA,name="PSRL")
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.old_episode = 0
		self.t_episode = 0
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.policy = np.zeros((self.nU * self.nS, self.nA))
		self.u = np.zeros(self.nU * self.nS)

		#self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		#self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))

		#self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM = RM


	def mapping(self, u, s):
		return u * self.nS + s



	def name(self):
		return "TSDE(CP)"

	# To reinitialize the learner with a given initial state inistate.
	def reset(self, inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.u = np.zeros(self.nU * self.nS)
		self.policy = np.zeros((self.nU * self.nS, self.nA))

		#self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		#self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))


		#self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM.reset()
		self.new_episode()


	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def VI(self, epsilon=0.01, max_iter=1000):

		u0 = self.u - min(self.u)
		u1 = np.zeros(self.nU * self.nS)
		itera = 0

		while True:
			for s_rm in range(self.nU):
				for s_mdp in range(self.nS):
					s = self.mapping(s_rm, s_mdp)
					temp = np.zeros(self.nA)
					for a in range(self.nA):
						# print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
						p = self.p_sampled[s, a]  # Allowed to sum  to <=1
						# print("Max_p of ",s,a, " : ", max_p)
						event = self.RM.events[s_mdp, a]
						if event != None:
							reward =  self.RM.rewards[s_rm, event]
						else:
							reward = 0
						temp[a] = reward + sum([u0[ns] * p[ns] for ns in range(self.nS)])

					# This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
					(u1[s], arg) = allmax(temp)
					nn = [-self.Nk[s, a] for a in arg]
					(nmax, arg2) = allmax(nn)
					choice = [arg[a] for a in arg2]
					self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

			diff = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1 - min(u1)
				break
			elif itera > max_iter:
				self.u = u1 - min(u1)
				print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
				break
			else:
				u0 = u1 - min(u1)
				u1 = np.zeros(self.nU * self.nS)
				itera += 1

	def new_episode(self):
		self.sumratios = 0.
		self.updateN()
		
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					#self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
					p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
					p=p[0]
					self.p_sampled[s,a] = [p[ns] for ns in range(self.nU * self.nS)]


		self.VI(epsilon=1. / max(1, self.t))

	###### Steps and updates functions ######

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		self.Nkmax = 0.
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]
					self.Nkmax = max(self.Nkmax, self.Nk[s, a])
					self.vk[s, a] = 0

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
		self.t_episode += 1
		# if self.sumratios >= 1.:  # Stoppping criterion
		if (self.vk[s, action] >= max([1, self.Nk[s, action]])) or (self.t_episode > self.old_episode):  # Stopping criterion
			self.old_episode = self.t_episode
			self.t_episode = 0
			self.new_episode()
			action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
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

		#self.r_successCounts[s,action] += reward
		#self.r_failureCounts[s,action] += 1.-reward

		self.p_pseudoCounts[s,action,o] +=1

		self.t += 1







class TSDE_RM(Agent):
	def __init__(self, nS, nA, RM, delta):
		Agent.__init__(self, nS, nA,name="PSRL")
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.old_episode = 0
		self.t_episode = 0
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.Nkmax = 0
		self.policy = np.zeros((self.nU, self.nS, self.nA))
		self.u = np.zeros((self.nU, self.nS))

		self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))

		self.p_sampled = np.zeros((self.nU, self.nS, self.nA, self.nS))

		self.RM = RM

	def name(self):
		return "TSDE-SA"

	# To reinitialize the learner with a given initial state inistate.
	def reset(self, inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nS, self.nA))
		self.Nk = np.zeros((self.nS, self.nA))
		self.Nkmax = 0
		self.u = np.zeros((self.nU, self.nS))
		self.policy = np.zeros((self.nU, self.nS, self.nA))

		self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))


		self.p_sampled = np.zeros((self.nU, self.nS, self.nA, self.nS))

		self.new_episode()



	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def VI(self, epsilon=0.01, max_iter=1000):
		
		min_u = min([min(u) for u in self.u])
		u0 = np.array([u - min_u for u in self.u])
		u1 = np.zeros((self.nU, self.nS))
		itera = 0

		while True:
			for s_rm in range(self.nU):
				for s in range(self.nS):
					temp = np.zeros(self.nA)
					for a in range(self.nA):
						# print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
						p = self.p_sampled[s_rm, s, a]  # Allowed to sum  to <=1
						# print("Max_p of ",s,a, " : ", max_p)
						event = self.RM.events[s, a]
						if event != None:
							ns_rm = self.RM.transitions[s_rm, event]
							reward = self.RM.rewards[s_rm, event]
						else:
							ns_rm = s_rm
							reward = 0
						temp[a] = reward + sum([u0[ns_rm, ns] * p[ns] for ns in range(self.nS)])

					# This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
					(u1[s_rm, s], arg) = allmax(temp)
					nn = [-self.Nk[s, a] for a in arg]
					(nmax, arg2) = allmax(nn)
					choice = [arg[a] for a in arg2]
					self.policy[s_rm, s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

			diff  = [abs(x - y) for (x, y) in zip(u1[s_rm], u0[s_rm])]
			if (max(diff) - min(diff)) < epsilon:
				min_u = min([min(u) for u in u1])
				self.U = np.array([u - min_u for u in u1])
				break
			elif itera > max_iter:
				min_u = min([min(u) for u in u1])
				self.U = np.array([u - min_u for u in u1])
				print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
				break
			else:
				min_u = min([min(u) for u in u1])
				u0 = np.array([u - min_u for u in u1])
				u1 = np.zeros(self.nS)
				itera += 1

	def new_episode(self):
		self.sumratios = 0.
		self.updateN()

		for s_rm in range(self.nU):
			for s in range(self.nS):
				for a in range(self.nA):
					p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
					p=p[0]
					self.p_sampled[s_rm, s,a] = [p[ns] for ns in range(self.nS)]


		self.VI(epsilon=1. / max(1, self.t))

	###### Steps and updates functions ######

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		self.Nkmax = 0.
		for s in range(self.nS):
			for a in range(self.nA):
				self.Nk[s, a] += self.vk[s, a]
				self.Nkmax = max(self.Nkmax, self.Nk[s, a])
				self.vk[s, a] = 0

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		action = categorical_sample([self.policy[RM_state, state, a] for a in range(self.nA)], np.random)
		self.t_episode += 1
		# if self.sumratios >= 1.:  # Stoppping criterion
		if (self.vk[state, action] >= max([1, self.Nk[state, action]])) or (self.t_episode > self.old_episode):  # Stopping criterion
			self.old_episode = self.t_episode
			self.t_episode = 0
			self.new_episode()
			action = categorical_sample([self.policy[RM_state, state, a] for a in range(self.nA)], np.random)
		return action

	# To update the learner after one step of the current policy.
	def update(self, state, action, reward, observation):
		self.vk[state, action] += 1
		self.observations[0].append(observation)
		self.observations[1].append(action)
		self.observations[2].append(reward)

		self.p_pseudoCounts[state,action,observation] +=1

		self.t += 1






class TSDE_CP_unknown(Agent):
	def __init__(self, nS, nA, RM, delta):
		Agent.__init__(self, nS, nA,name="PSRL")
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.old_episode = 0
		self.t_episode = 0
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.policy = np.zeros((self.nU * self.nS, self.nA))
		self.u = np.zeros(self.nU * self.nS)

		self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM = RM


	def mapping(self, u, s):
		return u * self.nS + s



	def name(self):
		return "TSDE(CP)_R_unknown"

	# To reinitialize the learner with a given initial state inistate.
	def reset(self, inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.u = np.zeros(self.nU * self.nS)
		self.policy = np.zeros((self.nU * self.nS, self.nA))

		self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))


		self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM.reset()
		self.new_episode()


	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def VI(self, epsilon=0.01, max_iter=1000):

		u0 = self.u - min(self.u)
		u1 = np.zeros(self.nU * self.nS)
		itera = 0

		while True:
			for s_rm in range(self.nU):
				for s_mdp in range(self.nS):
					s = self.mapping(s_rm, s_mdp)
					temp = np.zeros(self.nA)
					for a in range(self.nA):
						# print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
						p = self.p_sampled[s, a]  # Allowed to sum  to <=1
						# print("Max_p of ",s,a, " : ", max_p)
						temp[a] = self.r_sampled[s, a] + sum([u0[ns] * p[ns] for ns in range(self.nS)])

					# This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
					(u1[s], arg) = allmax(temp)
					nn = [-self.Nk[s, a] for a in arg]
					(nmax, arg2) = allmax(nn)
					choice = [arg[a] for a in arg2]
					self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

			diff = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1 - min(u1)
				break
			elif itera > max_iter:
				self.u = u1 - min(u1)
				print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
				break
			else:
				u0 = u1 - min(u1)
				u1 = np.zeros(self.nU * self.nS)
				itera += 1

	def new_episode(self):
		self.sumratios = 0.
		self.updateN()
		
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
					p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
					p=p[0]
					self.p_sampled[s,a] = [p[ns] for ns in range(self.nU * self.nS)]


		self.VI(epsilon=1. / max(1, self.t))

	###### Steps and updates functions ######

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		self.Nkmax = 0.
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]
					self.Nkmax = max(self.Nkmax, self.Nk[s, a])
					self.vk[s, a] = 0

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
		self.t_episode += 1
		# if self.sumratios >= 1.:  # Stoppping criterion
		if (self.vk[s, action] >= max([1, self.Nk[s, action]])) or (self.t_episode > self.old_episode):  # Stopping criterion
			self.old_episode = self.t_episode
			self.t_episode = 0
			self.new_episode()
			action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
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

		self.r_successCounts[s,action] += reward
		self.r_failureCounts[s,action] += 1.-reward

		self.p_pseudoCounts[s,action,o] +=1

		self.t += 1
























class PSRL_CP(Agent):
	def __init__(self, nS, nA, RM, delta):
		Agent.__init__(self, nS, nA,name="PSRL")
		self.nU = RM.nb_states
		self.nS = nS
		self.nA = nA
		self.old_episode = 0
		self.t_episode = 0
		self.t = 1
		self.delta = delta
		self.observations = [[], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.policy = np.zeros((self.nU * self.nS, self.nA))
		self.u = np.zeros(self.nU * self.nS)

		self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM = RM


	def mapping(self, u, s):
		return u * self.nS + s



	def name(self):
		return "PSRL(CP)"

	# To reinitialize the learner with a given initial state inistate.
	def reset(self, inistate):
		self.t = 1
		self.observations = [[inistate], [], []]
		self.vk = np.zeros((self.nU * self.nS, self.nA))
		self.Nk = np.zeros((self.nU * self.nS, self.nA))
		self.Nkmax = 0
		self.u = np.zeros(self.nU * self.nS)
		self.policy = np.zeros((self.nU * self.nS, self.nA))

		self.r_successCounts = np.ones((self.nU * self.nS, self.nA))
		self.r_failureCounts = np.ones((self.nU * self.nS, self.nA))
		self.p_pseudoCounts = np.ones((self.nU * self.nS, self.nA, self.nU * self.nS))


		self.r_sampled = np.zeros((self.nU * self.nS, self.nA))
		self.p_sampled = np.zeros((self.nU * self.nS, self.nA, self.nU * self.nS))

		self.RM.reset()
		self.new_episode()


	# The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
	def VI(self, epsilon=0.01, max_iter=1000):

		u0 = self.u - min(self.u)
		u1 = np.zeros(self.nU * self.nS)
		itera = 0

		while True:
			for s_rm in range(self.nU):
				for s_mdp in range(self.nS):
					s = self.mapping(s_rm, s_mdp)
					temp = np.zeros(self.nA)
					for a in range(self.nA):
						# print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
						p = self.p_sampled[s, a]  # Allowed to sum  to <=1
						# print("Max_p of ",s,a, " : ", max_p)
						temp[a] = self.r_sampled[s, a] + sum([u0[ns] * p[ns] for ns in range(self.nS)])

					# This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
					(u1[s], arg) = allmax(temp)
					nn = [-self.Nk[s, a] for a in arg]
					(nmax, arg2) = allmax(nn)
					choice = [arg[a] for a in arg2]
					self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

			diff = [abs(x - y) for (x, y) in zip(u1, u0)]
			if (max(diff) - min(diff)) < epsilon:
				self.u = u1 - min(u1)
				break
			elif itera > max_iter:
				self.u = u1 - min(u1)
				print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
				break
			else:
				u0 = u1 - min(u1)
				u1 = np.zeros(self.nU * self.nS)
				itera += 1

	def new_episode(self):
		self.sumratios = 0.
		self.updateN()
		
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
					p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
					p=p[0]
					self.p_sampled[s,a] = [p[ns] for ns in range(self.nU * self.nS)]


		self.VI(epsilon=1. / max(1, self.t))

	###### Steps and updates functions ######

	# Auxiliary function to update N the current state-action count.
	def updateN(self):
		self.Nkmax = 0.
		for s_rm in range(self.nU):
			for s_mdp in range(self.nS):
				s = self.mapping(s_rm, s_mdp)
				for a in range(self.nA):
					self.Nk[s, a] += self.vk[s, a]
					self.Nkmax = max(self.Nkmax, self.Nk[s, a])
					self.vk[s, a] = 0

	# To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
	def play(self, state):
		RM_state = self.RM.current_state
		s = self.mapping(RM_state, state)
		action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
		# if self.sumratios >= 1.:  # Stoppping criterion
		if (self.vk[s, action] >= max([1, self.Nk[s, action]])):  # Stopping criterion
			self.old_episode = self.t_episode
			self.new_episode()
			action = categorical_sample([self.policy[s, a] for a in range(self.nA)], np.random)
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

		self.r_successCounts[s,action] += reward
		self.r_failureCounts[s,action] += 1.-reward

		self.p_pseudoCounts[s,action,o] +=1

		self.t += 1