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
        
        remaining food (newFood) 
        and 
        Pacman position after moving (newPos).

        newScaredTimes holds the number of moves that each ghost will remain
        scared 
        
        because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        distanceToFood = []
        indexOfFood = []

        # distance of pac to food items

        for food in newFood.asList():
            distanceToFood.append(manhattanDistance(food, newPos))
            indexOfFood.append(food)

        if distanceToFood:
            minimum_food_dist = min(distanceToFood)
        else:
            minimum_food_dist = 0

        updated_score = successorGameState.getScore()

        #magnitude of food distance to score

        updated_score += 1/(minimum_food_dist + 0.5)


        newGhostStates = successorGameState.getGhostStates()

        positions_of_ghost = []

        for ghostState in newGhostStates:
            ghost = ghostState.configuration.pos
            positions_of_ghost.append(ghost)

        distanceToGhost = []
        indexOfGhost = []

        # distance of pac to ghost

        for pos_ghost in positions_of_ghost:
            distanceToGhost.append(manhattanDistance(pos_ghost, newPos))
            indexOfGhost.append(pos_ghost)

        if distanceToGhost:
            minimum_ghost_dist = min(distanceToGhost)
        else:
            minimum_ghost_dist = 0

        #magnitude of ghost distance to score

        updated_score -= 1/(minimum_ghost_dist  - 0.8)


        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        if newScaredTimes:
            check_scared = max(newScaredTimes)
        
        if check_scared > 0:
            updated_score += 100


        "*** YOUR CODE HERE ***"
        return updated_score

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

    def maximize(self, gameState, depth, ghostInfo):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True

        # end or depth 0
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        PossibleActionsForAgent = gameState.getLegalActions()

        evaluated_values = []

        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)

            # calling minimizer for opponents 

            eval = self.minimize(successor, depth, ghostInfo)
            evaluated_values.append(eval)

        return max(evaluated_values)



    def minimize(self, gameState, depth, ghostInfo):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True

        Ghost_Index = (gameState.getNumAgents() - 1) - ghostInfo + 1 
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        PossibleActionsForGhost = gameState.getLegalActions(Ghost_Index)

        evaluated_values = []

        for action in PossibleActionsForGhost:
            successor = gameState.generateSuccessor(Ghost_Index, action)

            if ghostInfo > 1:

                # calling minimizer for all opponents 

                eval = self.minimize(successor, depth, ghostInfo-1)
            else:

                # calling maximizer again for pacman 

                eval = self.maximize(successor, depth-1, ((gameState.getNumAgents() - 1)))

            evaluated_values.append(eval)

        return min(evaluated_values)


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

        PossibleActionsForAgent = gameState.getLegalActions()

        evaluated_values = []

        PossibleActions = PossibleActionsForAgent

        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)
            # driver function for evaluating states for pacman
            eval = self.minimize(successor, self.depth, (gameState.getNumAgents() - 1))
            evaluated_values.append(eval)

        return PossibleActions[evaluated_values.index(max(evaluated_values))]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maximize(self, gameState, depth, ghostInfo, alpha, beta):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        PossibleActionsForAgent = gameState.getLegalActions()

        eval_max = - float('inf')


        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)
            # calling minimizer for opponents 
            eval = self.minimize(successor, depth, ghostInfo, alpha, beta)
            eval_max = max(eval_max, eval)

            # comparing max for pruning

            if eval_max > beta:
                break
            alpha = max(eval_max, alpha)

        return eval_max



    def minimize(self, gameState, depth, ghostInfo, alpha, beta):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True

        Ghost_Index = (gameState.getNumAgents() - 1) - ghostInfo + 1 
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        eval_min = float('inf')

        PossibleActionsForGhost = gameState.getLegalActions(Ghost_Index)


        for action in PossibleActionsForGhost:
            successor = gameState.generateSuccessor(Ghost_Index, action)

            if ghostInfo > 1:
                # calling minimizer for all opponents 
                eval = self.minimize(successor, depth, ghostInfo-1, alpha, beta)
            else:
                # calling maximizer for pacman again
                eval = self.maximize(successor, depth-1, ((gameState.getNumAgents() - 1)), alpha, beta)

            eval_min = min(eval_min, eval)

            # comparing min for prune update

            if eval_min < alpha:
                break
            beta = min(eval_min, beta)

        return eval_min

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = - float('inf')
        beta = float('inf')
        eval_max = - float('inf')
        
        PossibleActionsForAgent = gameState.getLegalActions()

        evaluated_values = []

        PossibleActions = PossibleActionsForAgent

        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)
            # driver function 
            eval = self.minimize(successor, self.depth, (gameState.getNumAgents() - 1), alpha, beta)
            eval_max = max(eval_max, eval)

            if eval_max > beta:
                return eval_max
            alpha = max(eval_max, alpha)

            evaluated_values.append(eval_max)
            

        return PossibleActions[evaluated_values.index(max(evaluated_values))]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def maximize(self, gameState, depth, ghostInfo):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        PossibleActionsForAgent = gameState.getLegalActions()

        evaluated_values = []

        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)
            # calling minimzer for opponents
            eval = self.minimize(successor, depth, ghostInfo)
            evaluated_values.append(eval)

        return max(evaluated_values)



    def minimize(self, gameState, depth, ghostInfo):

        Winning_Check = gameState.isWin()
        Losing_Check = gameState.isLose()
        Depth_Check = False

        if depth == 0:
            Depth_Check =  True

        Ghost_Index = (gameState.getNumAgents() - 1) - ghostInfo + 1 
        
        if Winning_Check or Losing_Check or Depth_Check:
            return self.evaluationFunction(gameState)


        PossibleActionsForGhost = gameState.getLegalActions(Ghost_Index)

        evaluated_values = []

        for action in PossibleActionsForGhost:
            successor = gameState.generateSuccessor(Ghost_Index, action)

            if ghostInfo > 1:
                eval = self.minimize(successor, depth, ghostInfo-1)
            else:
                eval = self.maximize(successor, depth-1, ((gameState.getNumAgents() - 1)))

            evaluated_values.append(eval)

            # averaging out for expected values
        
        avg = sum(evaluated_values)/(len(PossibleActionsForGhost))

        return avg

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        PossibleActionsForAgent = gameState.getLegalActions()

        evaluated_values = []

        PossibleActions = PossibleActionsForAgent

        for action in PossibleActionsForAgent:
            successor = gameState.generateSuccessor(self.index, action)
            # driver
            eval = self.minimize(successor, self.depth, (gameState.getNumAgents() - 1))
            evaluated_values.append(eval)

        return PossibleActions[evaluated_values.index(max(evaluated_values))]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacman_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()


    ghost_states = currentGameState.getGhostStates()
    ghost_score = 0

    ghostPosition = ghost_states[0].getPosition()
    # distance from ghost
    pac_ghost_dist = manhattanDistance(pacman_pos, ghostPosition)

    near_food = 0

    if food_pos:
        # distance from food
      near_food = manhattanDistance(food_pos[0], pacman_pos)

    if ghost_states[0].scaredTimer > 1:
        # adding magnitude of ghost distance based on scared Timer
        ghost_score += 100 + pac_ghost_dist

    foodcount = currentGameState.getNumFood()

    # magnitude based on the game

    total_score = (currentGameState.getScore()) - 18 * foodcount + ghost_score - 1.5 * near_food

    return total_score
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
