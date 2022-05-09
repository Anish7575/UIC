# Name: Sai Anish Garapati
# UIN: 650208577

import math
import time
import tracemalloc


# Class for creating a puzzle node in the Breadth first search tree
# The puzzle nodes keeps track of current state, action responsible for this state,
# parent state and position of empty tile

class PuzzleNode:
    def __init__(self, state, action, parent, empty_pos):
        self.state = state
        self.action = action
        self.parent = parent
        self.empty_pos = empty_pos


# Class for initializing the puzzle board and solving the board to reach the goal state

class PuzzleSolver:
    def __init__(self, rootNode):
        self.rootNode = rootNode
        self.frontier = list()
        self.explored_set = set()
        self.moves = ['U', 'R', 'D', 'L']

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

    # Check if the node state is already in explored set or frontier list
    def checkChildNodeState(self, childNode):
        if tuple(childNode.state) in self.explored_set:
            return False
        if childNode.state in [node.state for node in self.frontier]:
            return False
        return True

    # BFS alogorithm for searching the solution and returning the set of actions
    def breadthFirstSearch(self):
        if self.goalCheck(self.rootNode):
            return True

        self.frontier.append(self.rootNode)

        # Search using the BFS until the frontier list is not empty
        while len(self.frontier) > 0:
            expandNode = self.frontier[0]
            self.frontier.pop(0)

            # Add expanded nodes into the explored_set for fast lookup
            self.explored_set.add(tuple(expandNode.state))

            for action in self.moves:
                childNode = self.computeChildNode(expandNode, action)
                if childNode.action == None:
                    continue
                
                # If the goal state is found, then backtrack to get the moves excuted to reach this state
                if self.goalCheck(childNode):
                    result_actions = []
                    while childNode.parent != None:
                        result_actions.append(childNode.action)
                        childNode = childNode.parent
                    result_actions.reverse()
                    return result_actions, len(self.explored_set)

                if self.checkChildNodeState(childNode):
                    self.frontier.append(childNode)


if __name__ == '__main__':
    tracemalloc.start()
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
                          parent=None, empty_pos=empty_pos)

    PuzzleSolver = PuzzleSolver(rootNode=rootNode)

    resultMoves, nodesExpanded = PuzzleSolver.breadthFirstSearch()
    end = time.time()

    print('\n')
    print('Moves: ', ' '.join(resultMoves))
    print('Number of Nodes expanded: ', nodesExpanded)
    print('Time Taken: ', str(end - start) + ' secs')
    print('Memory Used: ', str(tracemalloc.get_traced_memory()[1]/1024) + ' KB')
    print('\n')
    tracemalloc.stop()
