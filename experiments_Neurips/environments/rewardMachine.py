

# A reward machine is defined by a number of states (states inditified by int),  a list of high level events (we chose to use
# use the number of events and identify them as int), a list of transtions between states (matrix statexevent=next_state) and a
# list of rewards (matrix statexstate=rewards) that can be given when a transtion occurs.
class RewardMachine:
    def __init__(self, Events, Transitions, Rewards, init = 0):
        self.nb_states = len(Transitions)
        self.events = Events # SxAxS matrix with corresponding event or None
        self.transitions = Transitions
        self.rewards = Rewards
        self.current_state = init
        self.previous_state = init
        self.init = init

    def next_step(self, event):
        reward = 0
        self.previous_state = self.current_state
        if event != None:
            old_state = self.current_state
            self.current_state = self.transitions[self.current_state, event]
            reward = self.rewards[old_state, self.current_state]
        return reward

    def reset(self):
        self.previous_state = self.current_state
        self.current_state = self.init






class RewardMachine_s:
    def __init__(self, Events, Transitions, Rewards, init = 0):
        self.nb_states = len(Transitions)
        self.events = Events # SxAxS matrix with corresponding event or None
        self.transitions = Transitions
        self.rewards = Rewards
        self.current_state = init
        self.previous_state = init
        self.init = init

    def next_step(self, event):
        reward = 0
        self.previous_state = self.current_state
        if event != None:
            old_state = self.current_state
            self.current_state = self.transitions[self.current_state, event]
            reward = self.rewards[old_state, self.current_state]
        return reward

    def reset(self):
        self.previous_state = self.current_state
        self.current_state = self.init