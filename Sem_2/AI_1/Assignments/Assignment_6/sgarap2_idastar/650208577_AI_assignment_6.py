# Name: Sai Anish Garapati
# UIN: 650208577

import math
import time
from collections import deque
import sys

# Class for creating a puzzle node in the A* search tree
# The puzzle nodes keeps track of current state, action responsible for this state,
# parent node, position of empty tile and depth of the node

class PuzzleNode:
    def __init__(self, state, action, parent, empty_pos):
        self.state = state
        self.action = action
        self.parent = parent
        self.empty_pos = empty_pos
        if parent == None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1


# Class for initializing the puzzle board and solving the board to reach the goal state

class PuzzleSolver:
    def __init__(self, rootNode):
        self.rootNode = rootNode
        self.frontier = deque([])
        self.explored_nodes = 0
        self.moves = ['U', 'R', 'L', 'D']

    # Method to get the number of misplaced tiles from the board in current state
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 0]
    # Element at index i of the list should be (i + 1) except the last element
    def checkMisplacedTiles(self, node):
        count = 0
        for i in range(len(node.state)):
            if node.state[i] == 0:
                continue
            if node.state[i] != (i + 1):
                count += 1
        return count

    # Method to get the manhatten distance of the board from the goal state
    # This is the sum of the number of squares from desired location for eery tile
    # (i//rows, i%rows) is location of tile in 2D board
    # (node.state[i]//rows, node.state[i]%rows) is location of tile in 2D board for goal state
    def checkManhatten(self, node):
        dist = 0
        rows = int(math.sqrt(len(node.state)))
        for i in range(len(node.state)):
            if node.state[i] == 0:
                continue
            dist += abs(i//rows - (node.state[i] - 1)//rows) + abs(i%rows - (node.state[i] - 1)%rows)
        return dist

    # Compute the childNode from the current node, that is derived from the action applied
    def computeChildNode(self, node, action):
        row = node.empty_pos[0]
        col = node.empty_pos[1]
        numRows = int(math.sqrt(len(node.state)))

        childNode = PuzzleNode(state=node.state.copy(),
                               action=None, parent=node, empty_pos=None)

        # Using 2D index [row, col] as row*numRows + col in 1D list
        index = row*numRows + col

        # Checking if actions can be applied and swapping the elements accordingly
        if action == 'U' and row > 0:
            childNode.action = 'U'
            childNode.state[index], childNode.state[index - numRows] =\
                childNode.state[index - numRows], childNode.state[index]
            row -= 1

        elif action == 'R' and col < numRows - 1:
            childNode.action = 'R'
            childNode.state[index], childNode.state[index + 1] =\
                childNode.state[index + 1], childNode.state[index]
            col += 1

        elif action == 'D' and row < numRows - 1:
            childNode.action = 'D'
            childNode.state[index], childNode.state[index + numRows] =\
                childNode.state[index + numRows], childNode.state[index]
            row += 1

        elif action == 'L' and col > 0:
            childNode.action = 'L'
            childNode.state[index], childNode.state[index - 1] =\
                childNode.state[index - 1], childNode.state[index]
            col -= 1

        # Updating the empty tile in childNode state after the action is executed
        childNode.empty_pos = tuple([row, col])
        return childNode

    # Check if the current childNode creates a cycle in the graph in which case we avoid exploring that node
    # to avoid redundant paths
    def checkCycle(self, childNode):
        temp = childNode.parent
        while temp != None:
            if temp.state == childNode.state:
                return True
            temp = temp.parent
        return False

    # Iterative deeping A star search to search on A star using different thresholds iteratively until solution is found
    def id_a_star_search(self, heuristic):
        if heuristic == 'misplaced_tiles':
            threshold = self.checkMisplacedTiles(self.rootNode)
        elif heuristic == 'manhatten':
            threshold = self.checkManhatten(self.rootNode)

        # If heuristic is 0, goal state is found in the root node and function returns
        if threshold == 0:
            return [], 0, 0

        while(True):
            result, new_threshold, resultMoves, nodesExpanded, max_memory = self.a_star_search(heuristic, threshold)
            # If goal state is found for the current threshold, function returns moves, count of expanded nodes and memory
            # If goal state is not found for current threshold, it is updated with new threshold which is the minimum of all f(n) which are greater than current threshold
            # If new threshold is infinity, failure is returned
            if result == 'FOUND':
                return resultMoves, nodesExpanded, max_memory
            elif result == 'NOT FOUND':
                if new_threshold >= sys.maxsize:
                    return None, None, None
                else:
                    threshold = new_threshold
                    self.explored_nodes = 0
                    self.frontier.clear()


    # A* search for searching the solution and returning the set of actions for a specific threshold
    def a_star_search(self, heuristic, threshold):
        new_threshold = sys.maxsize
        result = 'FOUND'

        self.frontier.append(self.rootNode)
        max_memory = 0
        # Search using the heuristic function in depth first fashion until the frontier list is not empty
        while len(self.frontier) > 0:
            max_memory = max(max_memory, sys.getsizeof(self.frontier))
            
            expandNode = self.frontier[-1]
            self.frontier.pop()

            self.explored_nodes += 1

            for action in self.moves:
                childNode = self.computeChildNode(expandNode, action)
                if childNode.action == None:
                    continue

                if heuristic == 'misplaced_tiles':
                    h_n = self.checkMisplacedTiles(childNode)
                elif heuristic == 'manhatten':
                    h_n = self.checkManhatten(childNode)

                # If Heuristic function is 0, goal state is found and we return actions, explored nodes and memory used
                if h_n == 0:
                    result_actions = []
                    while childNode.parent != None:
                        result_actions.append(childNode.action)
                        childNode = childNode.parent
                    result_actions.reverse()
                    return result, new_threshold, result_actions, self.explored_nodes, max_memory

                if not self.checkCycle(childNode):
                    f_n = childNode.depth + h_n
                    if f_n <= threshold:
                        self.frontier.append(childNode)
                    else:
                        new_threshold = min(new_threshold, f_n)

        # If stack becomes empty, result is not found for the current threshold and 'NOT FOUND' is returned as result along with new_threshold
        result = 'NOT FOUND'
        return result, new_threshold, None, None, None


if __name__ == '__main__':
    # Taking input from Command Line
    init_state = list(map(int, input('Enter initial state:').strip().split()))
    length = len(init_state)

    # Check if the entered input matrix is a square matrix
    if math.sqrt(length) != int(math.sqrt(length)):
        print('Entered state is not a square matrix')
        exit(0)

    rows = int(math.sqrt(length))

    # Searching the empty position (0, 0) from the initial input state
    empty_pos = None
    for i in range(length):
        if init_state[i] == 0:
            empty_pos = tuple([int(i/rows), i % rows])
            break

    rootNode = PuzzleNode(state=init_state, action=None,
                          parent=None, empty_pos=empty_pos)

    # Case 1: Solving the initial state for Heuristic function = Misplaced Tiles
    puzzleHeuristic1 = PuzzleSolver(rootNode=rootNode)
    
    start = time.time()
    resultMoves, nodesExpanded, max_memory = puzzleHeuristic1.id_a_star_search(heuristic='misplaced_tiles')
    end = time.time()

    # If no solution is found we return from here as there would be no solution even for a different Heuristic function
    if resultMoves == None:
        print ('The given initial state has no solution\n')
        exit
    else:
        print('\n')
        print('Heuristic Function: Misplaced Tiles')
        print('Moves: ', ' '.join(resultMoves))
        print('Number of Nodes expanded: ', nodesExpanded)
        print('Time Taken: ', str(end - start) + ' secs')
        print('Memory used: ', str(max_memory) + 'B')

    # Case 2: Solving the initial state for Heuristic function = Manhatten distance
    puzzleHeuristic2 = PuzzleSolver(rootNode=rootNode)
    start = time.time()
    resultMoves, nodesExpanded, max_memory = puzzleHeuristic2.id_a_star_search(heuristic='manhatten')
    end = time.time()

    print('\n')
    print('Heuristic Function: Manhatten distance')
    print('Moves: ', ' '.join(resultMoves))
    print('Number of Nodes expanded: ', nodesExpanded)
    print('Time Taken: ', str(end - start) + ' secs')
    print('Memory used: ', str(max_memory) + 'B')
