# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        remaining_iter = self.iterations
        states = self.mdp.getStates()
        is_terminal = self.mdp.isTerminal
        actions_at = self.mdp.getPossibleActions
        q_val_func = self.computeQValueFromValues
        
        while remaining_iter:
            vals = util.Counter()
            for state in states:
                if not is_terminal(state):
                    vals[state] = float('-inf')
                    for action in actions_at(state):
                        q_value = q_val_func(state, action)
                        vals[state] = max(q_value, vals[state])
            
            self.values = vals
            remaining_iter = remaining_iter - 1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        trans_prob = self.mdp.getTransitionStatesAndProbs
        reward = self.mdp.getReward
        D = self.discount

        vals = 0

        for t, p in trans_prob(state, action):
            iter_val = self.getValue(t)
            R = reward(state, action, t)
            vals += p * (R + (D * iter_val))

        return vals

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        actions_at = self.mdp.getPossibleActions

        action_vals = util.Counter()

        q_val = self.computeQValueFromValues

        for a in actions_at(state):
            action_vals[a] = q_val(state, a)

        action = action_vals.argMax()
        return action
        
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        remaining_iter = self.iterations
        states = self.mdp.getStates()
        is_terminal = self.mdp.isTerminal
        actions_at = self.mdp.getPossibleActions
        q_val_func = self.computeQValueFromValues

        i = 0
        
        while remaining_iter:

            vals = self.values.copy()
            state = states[i % len(states)]

            if not is_terminal(state):
                vals[state] = float('-inf')
                for action in actions_at(state):
                    q_value = q_val_func(state, action)
                    vals[state] = max(q_value, vals[state])
            
            self.values = vals
            i+=1
            remaining_iter = remaining_iter - 1

        "*** YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def runValueIteration(self):
        Q = util.PriorityQueue()

        remaining_iter = self.iterations
        states = self.mdp.getStates()
        is_terminal = self.mdp.isTerminal
        actions_at = self.mdp.getPossibleActions
        q_val = self.getQValue

        preds = {}

        trans_prob = self.mdp.getTransitionStatesAndProbs

        for state in states:
            if not is_terminal(state):
                for action in actions_at(state):
                    for ns, code in trans_prob(state, action):
                        if ns in preds:
                            preds[ns].add(state)
                        else:
                            preds[ns] = {state}
        
        for state in states:
            if not is_terminal(state):
                q_lst = [q_val(state, action) for action in actions_at(state)]
                q_opt = max(q_lst)
                sub = abs(self.values[state] - q_opt)
                Q.push(state, -sub)

        while remaining_iter:
            if not Q.isEmpty():
                state = Q.pop()
            else:
                break
          
            if not is_terminal(state):
                q_lst = [q_val(state, action) for action in actions_at(state)]
                q_opt = max(q_lst)
                self.values[state] = q_opt

            for pred in preds[state]:
                if not is_terminal(pred):
                    q_lst = [q_val(pred, action) for action in actions_at(pred)]
                    q_opt = max(q_lst)
                    sub = abs(self.values[pred] - q_opt)
                    if sub > self.theta:
                        Q.update(pred, -sub)
            remaining_iter-=1
        "*** YOUR CODE HERE ***"

