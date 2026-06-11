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
from typing import Any

from game import Agent
from pacman import GameState


class _TypedAgent(Agent):
    def getAction(self, state) -> Any:
        raise NotImplementedError


def _foodPositions(gameState):
    walls = gameState.getWalls()
    width = getattr(walls, "width")
    height = getattr(walls, "height")
    foodPositions = []
    for x in range(width):
        for y in range(height):
            if gameState.hasFood(x, y):
                foodPositions.append((x, y))
    return foodPositions


class ReflexAgent(_TypedAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        gameState = state
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if successorGameState.isWin():
            return float("inf")
        if successorGameState.isLose():
            return float("-inf")

        score = successorGameState.getScore()
        foodList = _foodPositions(successorGameState)

        if foodList:
            closestFood = min(
                manhattanDistance(newPos, foodPos) for foodPos in foodList
            )
            score += 1.0 / (closestFood + 1.0)
        score -= 4 * len(foodList)

        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                score += 2.0 / (ghostDistance + 1.0)
            else:
                if ghostDistance <= 1:
                    score -= 10
                score -= 1.0 / (ghostDistance + 1.0)

        if action == Directions.STOP:
            score -= 2

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(_TypedAgent):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, state):
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
        gameState = state
        numAgents = gameState.getNumAgents()

        def advance(agentIndex, depth):
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth += 1
            return nextAgent, nextDepth

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent, nextDepth = advance(agentIndex, depth)

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    successorValue = value(successor, nextAgent, nextDepth)
                    if successorValue > bestValue:
                        bestValue = successorValue
                return bestValue

            bestValue = float("inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                successorValue = value(successor, nextAgent, nextDepth)
                if successorValue < bestValue:
                    bestValue = successorValue
            return bestValue

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        nextAgent, nextDepth = advance(0, 0)
        bestScore = float("-inf")
        bestAction = legalActions[0]
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            successorScore = value(successor, nextAgent, nextDepth)
            if successorScore > bestScore:
                bestScore = successorScore
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        gameState = state
        numAgents = gameState.getNumAgents()

        def advance(agentIndex, depth):
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth += 1
            return nextAgent, nextDepth

        def value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent, nextDepth = advance(agentIndex, depth)

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    successorValue = value(successor, nextAgent, nextDepth, alpha, beta)
                    if successorValue > bestValue:
                        bestValue = successorValue
                    if bestValue > beta:
                        return bestValue
                    if bestValue > alpha:
                        alpha = bestValue
                return bestValue

            bestValue = float("inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                successorValue = value(successor, nextAgent, nextDepth, alpha, beta)
                if successorValue < bestValue:
                    bestValue = successorValue
                if bestValue < alpha:
                    return bestValue
                if bestValue < beta:
                    beta = bestValue
            return bestValue

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        alpha = float("-inf")
        beta = float("inf")
        nextAgent, nextDepth = advance(0, 0)
        bestScore = float("-inf")
        bestAction = legalActions[0]
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            successorScore = value(successor, nextAgent, nextDepth, alpha, beta)
            if successorScore > bestScore:
                bestScore = successorScore
                bestAction = action
            if bestScore > alpha:
                alpha = bestScore
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        gameState = state
        numAgents = gameState.getNumAgents()

        def advance(agentIndex, depth):
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth += 1
            return nextAgent, nextDepth

        def value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent, nextDepth = advance(agentIndex, depth)

            if agentIndex == 0:
                bestValue = float("-inf")
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    successorValue = value(successor, nextAgent, nextDepth)
                    if successorValue > bestValue:
                        bestValue = successorValue
                return bestValue

            totalValue = 0.0
            probability = 1.0 / len(legalActions)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                totalValue += probability * value(successor, nextAgent, nextDepth)
            return totalValue

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        nextAgent, nextDepth = advance(0, 0)
        bestScore = float("-inf")
        bestAction = legalActions[0]
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            successorScore = value(successor, nextAgent, nextDepth)
            if successorScore > bestScore:
                bestScore = successorScore
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Combines score with nearest-food pressure, penalties for remaining food/capsules, strong active-ghost avoidance, and rewards for reachable scared ghosts.
    """
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    position = currentGameState.getPacmanPosition()
    foodList = _foodPositions(currentGameState)
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if foodList:
        closestFood = min(manhattanDistance(position, foodPos) for foodPos in foodList)
        score += 6.0 / (closestFood + 1.0)
    score -= 4 * len(foodList)

    if capsuleList:
        closestCapsule = min(
            manhattanDistance(position, capsulePos) for capsulePos in capsuleList
        )
        score += 3.0 / (closestCapsule + 1.0)
    score -= 6 * len(capsuleList)

    activeGhostDistances = []
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(position, ghostPos)
        scaredTime = ghostState.scaredTimer

        if scaredTime > 0:
            score += 8.0 / (ghostDistance + 1.0)
            score += max(scaredTime - ghostDistance, 0)
        else:
            activeGhostDistances.append(ghostDistance)

    if activeGhostDistances:
        closestActiveGhost = min(activeGhostDistances)
        if closestActiveGhost <= 1:
            score -= 200
        else:
            score -= 2.0 / closestActiveGhost

    return score


# Abbreviation
better = betterEvaluationFunction
