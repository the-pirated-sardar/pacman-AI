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
        for iteration in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    maxValue = -999999
                    for action in self.mdp.getPossibleActions(state):
                        t = self.mdp.getTransitionStatesAndProbs(state, action)
                        value = 0
                        for nextState, prob in t:
                            value += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
                        if value > maxValue:
                            maxValue = value
                    if maxValue != -999999:
                        newValues[state] = maxValue
            self.values = newValues


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
        qValue = 0

        t = self.mdp.getTransitionStatesAndProbs(state, action)
        for stateNProb in t:
            qValue += stateNProb[1] * (self.mdp.getReward(state, action, stateNProb[1]) + self.discount * self.values[stateNProb[0]])
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #check terminal state
        if self.mdp.isTerminal(state):
          return None
        
        #get possible actions
        actions = self.mdp.getPossibleActions(state)
        
        #find best action by maximum corresponding qValue
        allActions = {}
        for action in actions:
            allActions[action] = self.computeQValueFromValues(state, action)

        return max(allActions, key=allActions.get)

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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for iteration in range(self.iterations):   
            state = states[iteration % len(states)]
            if not self.mdp.isTerminal(state):
               actions = self.mdp.getPossibleActions(state)
               # update value of state with maximum qValue
               self.values[state] = max([self.getQValue(state, action) for action in actions]) 

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
        "*** YOUR CODE HERE ***"

        # get predecessors
        predecessors = {}
        pQueue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for stateNProb in self.mdp.getTransitionStatesAndProbs(state, action):
                        if stateNProb[0] in predecessors:
                            predecessors[stateNProb[0]].add(state)
                        else:
                            predecessors[stateNProb[0]] = {state}

        # push states into priority queue
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                qValues = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                diff = abs(max(qValues) - self.values[state])
                pQueue.push(state, -diff)

        # update values
        for iteration in range(self.iterations):
            if pQueue.isEmpty():
                break
            state = pQueue.pop()
            if not self.mdp.isTerminal(state):
                qValues = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                self.values[state] = max(qValues)

            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    qValues = [self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)]
                    diff = abs(max(qValues) - self.values[p])
                    if diff > self.theta:
                        pQueue.update(p, -diff)

