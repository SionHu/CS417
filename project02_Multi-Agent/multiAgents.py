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

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        currPos = successorGameState.getPacmanPosition()
        currFoodList = currentGameState.getFood().asList() # store curr available food locations
        dist = -9999999 # value will be returned eventually

        if action == 'Stop': return dist

        # when the distance between food and agent is small, score gets rewarded
        for food in currFoodList:
            dist = -1 * (abs(currPos[0] - food[0]) + abs(currPos[1] - food[1]))

        # whne the ghost meats the pacman, score is bad
        for state in newGhostStates:
            if state.getPosition() == currPos:
                return -9999999 # returns extemly negative to kill this possibility

        return dist
        # return successorGameState.getScore()

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
    def getValue(self, gameState, depth, agentcounter):
        if agentcounter >= gameState.getNumAgents():
            depth += 1
            agentcounter = 0

        if (depth == self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        elif agentcounter == 0: # find the max value
            maximum = ["", -float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return self.evaluationFunction(gameState)

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = self.getValue(currState, depth, agentcounter + 1)
                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]
                if newVal > maximum[1]:
                    maximum = [action, newVal]
            return maximum

        else: # find the min value
            minimum = ["", float("inf")]
            actions = gameState.getLegalActions(agentcounter)

            if not actions:
                return self.evaluationFunction(gameState)

            for action in actions:
                currState = gameState.generateSuccessor(agentcounter, action)
                current = self.getValue(currState, depth, agentcounter + 1)
                if type(current) is not list:
                    newVal = current
                else:
                    newVal = current[1]
                if newVal < minimum[1]:
                    minimum = [action, newVal]
            return minimum

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
        """
        "*** YOUR CODE HERE ***"
        actionsList = self.getValue(gameState, 0, 0) # watch out for self counts as first argument
        return actionsList[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxNode(self, gameState, numGhosts, plyCounter, alpha, beta):
        if gameState.isWin() or gameState.isLose() or plyCounter == 0:
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions()
        v = - float('inf')
        for action in legalActions:
            successorState = gameState.generateSuccessor(self.index, action)
            v = max(v, self.minNode(successorState, numGhosts, plyCounter, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minNode(self, gameState, numGhosts, plyCounter, alpha, beta):
        if gameState.isWin() or gameState.isLose() or plyCounter == 0:
            return self.evaluationFunction(gameState)

        totalNumGhosts = gameState.getNumAgents() - 1
        currentGhostIndex = totalNumGhosts - numGhosts + 1
        legalActions = gameState.getLegalActions(currentGhostIndex)
        v = float('inf')
        if numGhosts > 1:
            for action in legalActions:
                successorState = gameState.generateSuccessor(currentGhostIndex, action)
                v = min(v, self.minNode(successorState, numGhosts - 1, plyCounter, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
        else:
            for action in legalActions:
                successorState = gameState.generateSuccessor(currentGhostIndex, action)
                v = min(v, self.maxNode(successorState, totalNumGhosts, plyCounter - 1, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        actions = []
        evaluations = []

        alpha = - float('inf')
        beta = float('inf')
        v = - float('inf')
        for action in gameState.getLegalActions():
            actions.append(action)
            numGhosts = gameState.getNumAgents() - 1
            successorState = gameState.generateSuccessor(self.index, action)
            v = max(v, self.minNode(successorState, numGhosts, self.depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)

            evaluations.append(v)

        maxEvaluationIndex = evaluations.index(max(evaluations))
        return actions[maxEvaluationIndex]

        # util.raiseNotDefined()

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

        pactions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        for action in pactions:
            v = self.getValue(gameState.generateSuccessor(0, action), 0, 0)
            if v > bestValue:
                bestAction = action
                bestValue = v

        return bestAction

    def getValue(self, state, prevAgentIndex, depth):
        agentIndex = (prevAgentIndex + 1) % state.getNumAgents()
        legalActions = state.getLegalActions(agentIndex)

        if depth == self.depth or len(legalActions) == 0:
            score = self.evaluationFunction(state)
            return score

        bestValue = 0.0

        if agentIndex == 0:
            "Max value, increment depth"
            bestValue = float("-inf")
            for action in legalActions:
                v = self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth)
                if v > bestValue:
                    bestValue = v
        else:
            "Do expected value"
            if (agentIndex + 1) == state.getNumAgents():
                depth += 1

            for action in legalActions:
                bestValue += self.getValue(state.generateSuccessor(agentIndex, action), agentIndex, depth) * 1.0
            if len(legalActions) > 0:
                bestValue /= len(legalActions)

        return bestValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Our heuristic works as follows, maximize score by running
      away from ghosts and getting as close as possible to the minimum food
      pellot at each state. To solve the pickle of being stuck between two food pellots
      (0   *Pacman*     0) we add an incrementing counter to the manhattanDistance
      between our agent's position and the distance to the food. We also add
      incentive for eating capsules if the capsule is closer than the ghost.
      Our heuristic averages ~1100 points.
    """
    foodCoeff, powerCoeff, ghostCoeff= 1,3,1,
    foodScore, powerScore, ghostScore= 0,0,0,

    currPos, currFood = currentGameState.getPacmanPosition(), currentGameState.getFood()
    currGhostPos, currPower =  currentGameState.getGhostPositions(), currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    isScared = True if max(scaredTimes)!=0 else False

    closestFood = float(min([currFood.width + currFood.height] + [util.manhattanDistance(currPos, foodPos) for foodPos in currFood.asList()]))
    closestGhost = float(min([util.manhattanDistance(currPos, ghostPos) for ghostPos in currGhostPos]))
    closestPow = float(min([len(currPower)] + [util.manhattanDistance(powerPos, currPos) for powerPos in currPower]))


    foodScore = 1 if len(currFood.asList())==0 else 1/closestFood
    powerScore = 1 if len(currPower)==0 else 1/closestPow
    ghostScore = -100 if closestGhost < 1 else 1/closestGhost #*100 if isScared else closestGhost

    if isScared and closestGhost < max(scaredTimes):
        ghostCoeff, ghostScore = 100, abs(ghostScore)

    return foodCoeff*foodScore + ghostCoeff*ghostScore + powerCoeff*powerScore + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
