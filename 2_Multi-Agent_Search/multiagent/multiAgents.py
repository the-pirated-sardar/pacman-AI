# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #get current game state and pacman position
        currentPos = currentGameState.getPacmanPosition()

        #check for best case and return highest score for winning state if true
        if successorGameState.isWin():
            return 99999
        #worst case if and when pacman position is same as ghost position, but ghost is not scared
        for ghost in newGhostStates:
            if ghost.getPosition() == currentPos and ghost.scaredTimer == 0:
                return -99999
        score = 0

        #if an action is stop, start deducting score to avoid stop action
        if action == Directions.STOP:
            score -= 100

        #get the food list and calculate the distance between pacman and food
        foodDistance = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        #increase score for food that is closer to pacman by inverse of distance
        score += 1.0/min(foodDistance)
        #decrease score for number of food left along a path by the path length/2
        score -= len(newFood.asList())*0.5

        #get the ghost list and calculate the distance between pacman and ghost
        ghostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhost = min(ghostDistances)
        #get the ghost list and calculate the distance between pacman and new ghost in new GameState
        newGhostDistances = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestNewGhost = min(newGhostDistances)
        
        #if ghost is far away, increase score by 100 and if ghost is near, decrease score by 100
        if nearestNewGhost < nearestGhost:
            score += 100
        else:
            score -= 100

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #getting all actions for pacman to maximize best moves
        actions = gameState.getLegalActions(0)
        #initialize an empty dictionary to store actions and corresponding values to return the maximum value action
        allActions = {}
        for action in actions:
            allActions[action] = self.minValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key=allActions.get)
        
    def minValue(self, state, depth, agentIndex):
        #get number of agents and legal actions for argument agentIndex
        numAgents = state.getNumAgents() #?
        actions = state.getLegalActions(agentIndex)

        #if legal actions are empty, return the evaluation function value
        if not actions:
            return self.evaluationFunction(state)
        
        #pacman will move last
        if agentIndex == numAgents - 1:
            minimumValue = min(self.maxValue(state.generateSuccessor(agentIndex, action), depth) for action in actions)
        else:
            minimumValue = min(self.minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1) for action in actions)

        return minimumValue
    
    #agentIndex is 0 for pacman, therefore agentIndex is set as a default argument
    def maxValue(self, state, depth, agentIndex=0):
        #get legal actions for argument agentIndex
        actions = state.getLegalActions(agentIndex)

        #if legal actions are empty or depth is reached, return the evaluation function value
        if not actions or depth == self.depth:
            return self.evaluationFunction(state)
        
        maximumValue = max(self.minValue(state.generateSuccessor(agentIndex, action), depth+1, agentIndex+1) for action in actions)

        return maximumValue


        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #getting all actions for pacman to maximize best moves
        actions = gameState.getLegalActions(0)
        #initialize an empty dictionary to store actions and corresponding values to return the maximum value action
        allActions = {}
        #initialize alpha and beta values
        alpha = -99999
        beta = 99999

        for action in actions:
            value = self.minValue(gameState.generateSuccessor(0, action), 1, alpha, beta, 1)
            allActions[action] = value
            if value > beta:
                return action
            alpha = max(alpha, value)

        return max(allActions, key=allActions.get)

    def minValue(self, state, depth, alpha, beta, agentIndex):
        #get number of agents and legal actions for argument agentIndex
        numAgents = state.getNumAgents()
        actions = state.getLegalActions(agentIndex)

        #if legal actions are empty, return the evaluation function value
        if not actions:
            return self.evaluationFunction(state)
        
        #initialize values
        minimumValue = 99999
        currentBeta = beta

        #pacman will move last
        if agentIndex == numAgents - 1:
            for action in actions:
                minimumValue = min(minimumValue, self.maxValue(state.generateSuccessor(agentIndex, action), depth, alpha, currentBeta))
                if minimumValue < alpha:
                    return minimumValue
                currentBeta = min(currentBeta, minimumValue)
        else:
            for action in actions:
                minimumValue = min(minimumValue, self.minValue(state.generateSuccessor(agentIndex, action), depth, alpha, currentBeta, agentIndex + 1))
                if minimumValue < alpha:
                    return minimumValue
                currentBeta = min(currentBeta, minimumValue)

        return minimumValue
    
    #agentIndex is 0 for pacman, therefore agentIndex is set as a default argument
    def maxValue(self, state, depth, alpha, beta, agentIndex=0):
        #get legal actions for argument agentIndex
        actions = state.getLegalActions(agentIndex)

        #if legal actions are empty or depth is reached, return the evaluation function value
        if not actions or depth == self.depth:
            return self.evaluationFunction(state)
        
        #initialize values
        maximumValue = -99999
        currentAlpha = alpha

        for action in actions:
            maximumValue = max(maximumValue, self.minValue(state.generateSuccessor(agentIndex, action), depth+1, currentAlpha, beta, agentIndex+1))
            if maximumValue > beta:
                return maximumValue
            currentAlpha = max(currentAlpha, maximumValue)

        return maximumValue
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #getting all actions for pacman to maximize best moves
        actions = gameState.getLegalActions(0)
        #initialize an empty dictionary to store actions and corresponding values to return the maximum value action
        allActions = {}
        for action in actions:
            allActions[action] = self.expValue(gameState.generateSuccessor(0, action), 1, 1)

        return max(allActions, key=allActions.get)

    #function to calculate the expected depth
    def expValue(self, state, depth, agentIndex):
        numAgents = state.getNumAgents()
        actions = state.getLegalActions(agentIndex)

        if not actions:
            return self.evaluationFunction(state)

        expectedValue = 0
        probability = 1.0/len(actions) #uniform probability

        #pacman will move last
        for action in actions:
            if agentIndex == numAgents - 1:
                expectedValue += probability * self.maxValue(state.generateSuccessor(agentIndex, action), depth)
            else:
                expectedValue += probability * self.expValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

        return expectedValue
    
    def maxValue(self, state, depth, agentIndex=0):
        actions = state.getLegalActions(agentIndex)

        if not actions or depth == self.depth:
            return self.evaluationFunction(state)

        maxValue = max(self.expValue(state.generateSuccessor(agentIndex, action), depth+1, agentIndex+1) for action in actions)

        return maxValue

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I took inspiration from the original evaluation function and added a few more features to it. 
    The primary driver is to try to avoid the score from dropping, for which several parameters have been established.
    I assigned a score of 99999 for winning state and -99999 for losing state. I also assigned a score of -100 for stop action to avoid it. 
    I increased the score for closer food, farther ghost locations, ghost moving away, etc and vice versa.
    For eating food, the closer food is preferred by score and pellet location also increases score.
    For catching a scared ghost, the program considers the time left for the ghost to be scared and distance to decide to pursue the ghost or not.
    
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentCapsules = currentGameState.getCapsules()

    #check for best case and return highest score for winning state if true
    if currentGameState.isWin():
        return 99999
     #worst case if and when pacman position is same as ghost position, but ghost is not scared
    for ghost in currentGhostStates:
        if ghost.getPosition() == currentPos and ghost.scaredTimer == 0:
            return -99999

    score = 0

    #get the food list and calculate the distance between pacman and food
    foodDistance = [util.manhattanDistance(currentPos, food) for food in currentFood.asList()]
    #increase score for food that is closer to pacman by inverse of distance
    score += 1.0/min(foodDistance)

    if currentCapsules:
        #get the capsule list and calculate the distance between pacman and capsule
        capsuleDistance = [util.manhattanDistance(currentPos, capsule) for capsule in currentCapsules]
        #increase score for capsule that is closer to pacman by inverse of distance
        score += 1.0/min(capsuleDistance)
        #decrease score for number of capsule left along a path by the path length/2
        score -= len(currentCapsules)*0.5

    #if ghost is scared, get the ghost list and calculate the distance between pacman and ghost
    if currentScaredTimes[0] > 0:
        ghostDistances = [util.manhattanDistance(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
        nearestGhost = min(ghostDistances)
        #if ghost is far away, increase score by 100 and if ghost is near, decrease score by 100
        if nearestGhost > 3:
            score += 100
        else:
            score -= 100

    return currentGameState.getScore() + score
    

# Abbreviation
better = betterEvaluationFunction
