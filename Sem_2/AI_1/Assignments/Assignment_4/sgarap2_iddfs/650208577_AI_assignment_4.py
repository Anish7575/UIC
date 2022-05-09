# Name: Sai Anish Garapati
# UIN: 650208577

from collections import deque
import math
import time
import sys

# Class for creating a puzzle node in the Depth first search tree
# The puzzle nodes keeps track of current state, action responsible for this state,
# parent state, position of empty tile and the depth of the node in the tree

class PuzzleNode:
    def __init__(self, state, action, parent, nodeDepth, empty_pos):
        self.state = state
        self.action = action
        self.parent = parent
        self.nodeDepth = nodeDepth
        self.empty_pos = empty_pos

# Class for initializing the puzzle board and solving the board to reach the goal state

class PuzzleSolver:
    def __init__(self, rootNode):
        self.rootNode = rootNode
        self.explored_nodes = 0
        self.nodeStack = deque([])
        self.moves = ['L', 'D', 'R', 'U']

    # Method to check if the node contains the goal state
    # [1, 2, 3, 4, 5, 6, ,7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    # ELement at index i of the list should be (i + 1) except the last element
    def goalCheck(self, node):
        for i in range(len(node.state)):
            if i == len(node.state) - 1:
                if node.state[i] != 0:
                    return False
            elif node.state[i] != (i + 1):
                return False
        return True

    # Compute the childNode from the current node, that is derived from the action applied
    def computeChildNode(self, node, action):
        row = node.empty_pos[0]
        col = node.empty_pos[1]
        numRows = int(math.sqrt(len(node.state)))
        childNode = PuzzleNode(state=node.state.copy(), action=None, parent=node, nodeDepth=node.nodeDepth + 1, empty_pos=None)
        
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

    # Depth Fisrt Limited Search algorithm that implements DFS until it reaches a limit in the depth
    # In the case solution is found we return the actions taken, number of nodes expanded and memory usage
    # In the case we reach the depth limit we return 'cutoff' in which we implement the DFS with higher depth limit
    # In the case nodeStack becomes empty before reaching the depth limit with no solution then we return a 'failure'
    def DFSLimitedSearch(self, node, depth_limit):
        self.nodeStack.append(node)
        # Initialize the result with failure
        result = 'failure'
        max_memory = 0

        # Perform the search until the stack is not empty
        while len(self.nodeStack) > 0:
            max_memory = max(max_memory, len(self.nodeStack) * sys.getsizeof(node))
            node = self.nodeStack[-1]
            self.nodeStack.pop()
            # increment explored nodes counter
            self.explored_nodes += 1

            # If the goal state is found, then backtrack to get the moves excuted to reach this state
            if self.goalCheck(node):
                result_actions = []
                while node.parent != None:
                    result_actions.append(node.action)
                    node = node.parent
                result_actions.reverse()
                return result_actions, self.explored_nodes, max_memory
            else:
                if node.nodeDepth < depth_limit:
                    for action in self.moves:
                        childNode = self.computeChildNode(node, action)
                        if childNode.action == None:
                            continue
                        if not self.checkCycle(childNode):
                            self.nodeStack.append(childNode)
                else:
                    # Change result to cutoff in the case the tree depth reaches the set DFS limit
                    result = 'cutoff'
        # If no solution is found 'cutoff' or 'failur' is returned here
        return result, None, None

    # Iterative DFS with varying the depth limit for each iteration until we reach a solution or failure
    def iterativeDeepeningDFS(self):
        depth_limit = 0
        while (True):
            resultMoves, nodesExpanded, max_memory = self.DFSLimitedSearch(node=self.rootNode, depth_limit=depth_limit)
            if resultMoves == 'failure':
                return 'failure', None, None
            if resultMoves != 'cutoff':
                return resultMoves, nodesExpanded, max_memory
            depth_limit += 1
            self.explored_nodes = 0
            self.nodeStack.clear()


if __name__ == '__main__':
    # Taking input from Command Line
    init_state = list(map(int, input('Enter initial state:').strip().split()))

    start = time.time()

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
                          parent=None, nodeDepth=0, empty_pos=empty_pos)

    PuzzleSolver = PuzzleSolver(rootNode=rootNode)

    resultMoves, nodesExpanded, max_memory = PuzzleSolver.iterativeDeepeningDFS()
    end = time.time()

    print('\n')
    if resultMoves == 'failure':
        print('No solution exists for the given initial state')
    else:
        print('Moves: ', ' '.join(resultMoves))
        print('Number of Nodes expanded: ', nodesExpanded)
        print('Time Taken: ', str(end - start) + ' secs')
    print('Memory Used: ', str(max_memory) + 'B')
    print('\n')
