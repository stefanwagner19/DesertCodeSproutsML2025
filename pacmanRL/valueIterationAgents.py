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

        # for state in self.mdp.getStates():
        #     self.values[state] = 0.0
        for k in range(self.iterations):
            old_values = self.values.copy()

            for state in self.mdp.getStates():
                
                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                else:
                    max_value = None
                    
                    for action in self.mdp.getPossibleActions(state):
                        expected_reward = 0
                        
                        for (state_, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                            expected_reward += prob * (self.mdp.getReward(state, action, state_) + self.discount*old_values[state_])
                        
                        if max_value == None or expected_reward > max_value:
                            max_value = expected_reward

                    self.values[state] = max_value


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
        # util.raiseNotDefined()
        q_value = 0
        for (state_, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            q_value += prob * (self.mdp.getReward(state, action, state_) + self.discount*self.getValue(state_))
        return q_value


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        best_action = None
        max_value = None

        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)

            if max_value == None or q_value > max_value:
                max_value = q_value
                best_action = action

        return best_action


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
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
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for (state_, _) in self.mdp.getTransitionStatesAndProbs(state, action):

                    if state_ not in predecessors:
                        predecessors[state_] = set()
                    if not self.mdp.isTerminal(state_): 
                        predecessors[state_].add(state)

        queue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                current_value = self.getValue(state)
                max_q_value = self.getMaxQValue(state)
                diff = abs(current_value - max_q_value)
                queue.push(state, -diff)

        for k in range(self.iterations):

            if queue.isEmpty():
                return

            s = queue.pop()
            max_q_value = self.getMaxQValue(s)
            self.values[s] = max_q_value

            for p in predecessors[s]:
                current_value = self.getValue(p)
                max_q_value = self.getMaxQValue(p)
                diff = abs(current_value - max_q_value)

                if diff > self.theta:
                    queue.update(p, -diff)


    def getMaxQValue(self, state):
        max_q_value = None
        for action in self.mdp.getPossibleActions(state):
            
            q_value = self.getQValue(state, action)

            if max_q_value == None or q_value > max_q_value:
                max_q_value = q_value 

        return max_q_value