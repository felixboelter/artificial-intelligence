from sample_players import DataPlayer
import random
import math
import time

MAX_DEPTH = 20
MAX_TIME = 0.15 - 0.01  # Can change the time for the time limit you want to set


class UTCNode:
    def __init__(self, state, parent):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_actions = []
        self.parent = parent

    def add_child(self, state, action):
        """
        Create a new child and add that child to the self.children list
        and then action to the self.children_actions list
        """
        child = UTCNode(state, self)
        self.children.append(child)
        self.children_actions.append(action)

    def is_expanded(self):
        if len(self.children_actions) == len(self.state.actions()):
            return True
        return False


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):

        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            # Monte Carlo Tree Search
            self.queue.put(self.MCTS_search(state))
            # Alpha Beta Search
            #for depth in range(1, MAX_DEPTH + 1):
             #   self.queue.put(self.alpha_beta_search(state, depth))

    # Monte Carlo Tree Search
    def MCTS_search(self, state):
        """
        Search method of Monte Carlo Tree Search. Uses the Upper Confidence Bound for Trees (UCT) selection algorithm.
        Instead of iteration limits the function will try to have as many iterations as possible
        in the time limit set.
        Using pseudocode that is given here:
        https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=B7BB1338BDE1F287ECFC52AD86AFD055?doi=10.1.1.297.3086&rep=rep1&type=pdf#section.3
        """
        root_node = UTCNode(state, None)
        if root_node.state.terminal_test():
            return random.choice(state.actions())
        time_end = time.time() + MAX_TIME
        while time.time() < time_end:
            leaf_node = self.tree_policy(root_node)
            reward = self.default_policy(leaf_node.state)
            self.backprop(leaf_node, reward)
        # Return the action that leads to the best child of the root node
        return root_node.children_actions[root_node.children.index(self.best_child(root_node))]

    def tree_policy(self, node):
        """
        Select a leaf from the tree
        If the root node is not expanded, then expand the tree and create a new child
        If the root node is expanded, return the leaf with the best score.
        """
        while not node.state.terminal_test():
            if not node.is_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def best_child(self, node, c_factor=math.sqrt(2)):
        """
        Selection happens here by repeatedly selecting the child node
        with the highest (exploitation + c_factor * exploration) score
        Code inspired by:
        https://www.moderndescartes.com/essays/deep_dive_mcts/
        """
        return max(node.children, key=lambda child: self.exploitation(child) + c_factor * self.exploration(child))

    @staticmethod
    def exploitation(node):
        return node.reward / node.visits

    @staticmethod
    def exploration(node):
        return math.sqrt(math.log(node.parent.visits) / node.visits)

    @staticmethod
    def expand(node):
        """
        Expansion:
        choose a random untried action from
        all actions (state.actions()) - all explored actions (children_actions)
        Apply the action to the current game state and create a new game state
        Add a new child to the children list and add the action to the explored children_actions list
        """
        action = random.choice([a for a in node.state.actions() if a not in node.children_actions])
        new_state = node.state.result(action)
        node.add_child(new_state, action)
        return node.children[-1]

    @staticmethod
    def default_policy(state):
        """
        Simulation:
        Get the initializing player ID, choose a random move, apply the action until state becomes terminal
        Reward +1 if the agent holding initiative at the start of a simulation loses
        and -1 if the active agent when the simulation starts wins
        """
        player_id = state.player()
        while not state.terminal_test():
            action = random.choice(state.actions())
            state = state.result(action)
        if state._has_liberties(player_id):
            return -1
        else:
            return 1

    @staticmethod
    def backprop(node, reward):
        """
        Backup:
        Go back up the tree for each node increment the visit count
        and add the reward to the reward value
        Inverse the reward so that the exploitation value is the best
        from the perspective of the person whose turn it is to play
        """
        while node is not None:
            node.reward += reward
            node.visits += 1
            reward = -reward
            node = node.parent



    # Alpha Beta Search
    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.
        """
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():

            v = self.min_value(state.result(a), self.player_id, depth - 1, alpha, beta)
            alpha = max(alpha, v)
            if v >= best_score:
                best_score = v
                best_move = a
        return best_move

    def min_value(self, state, player_id, depth, alpha, beta):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """

        if state.terminal_test(): return state.utility(player_id)
        if depth == 0: return self.score(state, player_id)
        v = float("inf")

        for a in state.actions():
            v = min(v, self.max_value(state.result(a), player_id, depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def max_value(self, state, player_id, depth, alpha, beta):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if state.terminal_test(): return state.utility(player_id)
        if depth == 0: return self.score(state, player_id)
        v = float("-inf")

        for a in state.actions():
            v = max(v, self.min_value(state.result(a), player_id, depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    @staticmethod
    def score(state, player_id):
        my_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        my_moves = state.liberties(my_loc)
        opp_moves = state.liberties(opp_loc)
        return len(my_moves) - len(opp_moves)