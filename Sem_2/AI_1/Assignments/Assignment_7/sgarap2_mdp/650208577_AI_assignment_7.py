# Name: Sai Anish Garapati
# UIN: 650208577

import random

out_string = ''

# mdp class to create and store mdp board with information regarding size, terminal states,
# rewards, walls, discount factor and epsilon value

class mdp:
    def __init__(self, size, walls, terminal_states, reward, transition_probs, discount, epsilon):
        self.size = size
        self.board = [[reward] * self.size[1] for i in range(self.size[0])]
        for wall in walls:
            self.board[wall[0]][wall[1]] = 'X'
        self.terminal_states = []
        for state in terminal_states:
            self.terminal_states.append([state[0], state[1]])
            self.board[state[0]][state[1]] = state[2]
        self.probs = transition_probs
        self.discount = discount
        self.epsilon = epsilon
        self.actions = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        self.directions = ['U', 'R', 'D', 'L']
    
    # method to print the entire mdp
    def print_mdp(self):
        print(self.size)
        print(self.board)
        print(self.probs)
        print(self.discount)
        print(self.epsilon)

    # Method to print the Utility values in proper formatting
    def print_format(self, U):
        string = ''
        for r in U:
            for c in r:
                if isinstance(c, float):
                    string += '{:.12f}'.format(c)
                else:
                    string += c
                string += ' '
            string += '\n'
        string += '\n'
        return string
     
    # Method to calculate the optimal policy from the given maximum Utilities vector
    def get_optimal_policy(self, U):
        P = [['-'] * self.size[1] for i in range(self.size[0])]
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if [r, c] in self.terminal_states:
                    P[r][c] = 'T'
                    continue
                if self.board[r][c] == 'X':
                    continue
                max_val = float('-inf')
                max_idx = 0
                for i in range(len(self.actions)):
                    val = self.Q_value([r, c], i, U)
                    if val > max_val:
                        max_val = val
                        max_idx = i
                P[r][c] = self.directions[max_idx]
        return P

    # Get the policy in directions for a policy given in numbers
    def get_policy(self, P):
        P_dir = [['-'] * self.size[1] for i in range(self.size[0])]
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if [r, c] in self.terminal_states:
                    P_dir[r][c] = 'T'
                    continue
                if self.board[r][c] == 'X':
                    continue
                P_dir[r][c] = self.directions[P[r][c]]
        return P_dir
    
    # Method to retrieve the next state given the current state and action to be taken
    def get_next_state(self, state, action):
        new_state = [state[0] + action[0], state[1] + action[1]]
        if new_state[0] < 0 or new_state[0] >= self.size[0]:
            return state
        if new_state[1] < 0 or new_state[1] >= self.size[1]:
            return state
        if self.board[new_state[0]][new_state[1]] == 'X':
            return state
        return new_state
    
    # Method to calculate the Q value for a state given the action and current utlities
    def Q_value(self, state, action_idx, U):
        if state in self.terminal_states:
            return self.board[state[0]][state[1]]
        Q_val = 0
        
        for i, prob_pos in zip(range(action_idx - 1, action_idx + 3), [1, 0, 2, 3]):
            pos = i % len(self.actions)
            new_state = self.get_next_state(state, [self.actions[pos][0], self.actions[pos][1]])
            discount = self.discount
            if new_state in self.terminal_states:
                discount = 0
            Q_val += self.probs[prob_pos] * (self.board[new_state[0]][new_state[1]] + discount * U[new_state[0]][new_state[1]])
        return Q_val            
    
    # Function that computes value iteration on the given MDP and returns the maximum Utilities for the MDP
    def value_iteration(self):
        global out_string
        U_next = [[0.0] * self.size[1] for i in range(self.size[0])]
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.board[r][c] == 'X':
                    U_next[r][c] = '-' * 14

        iteration = 0        
        while True:
            U_cur = [row[:] for row in U_next]
            delta = 0
            out_string += ('Iteration: {}\n'.format(iteration))
            out_string += self.print_format(U_cur)
            for r in range(self.size[0]):
                for c in range(self.size[1]):
                    if self.board[r][c] == 'X':
                        continue
                    val = float('-inf')
                    for i in range(len(self.actions)):
                        val = max(val, self.Q_value([r, c], i, U_cur))
                    U_next[r][c] = val
                    if abs(U_next[r][c] - U_cur[r][c]) > delta:
                        delta = abs(U_next[r][c] - U_cur[r][c])
            
            if delta <= (self.epsilon * (1 - self.discount))/self.discount:
                break
            iteration += 1
        
        out_string += ('Utilities after final convergence:\n')
        out_string += self.print_format(U_cur)
        return U_cur
    
    # Method for policy evaluation to calculate utlities for current policy using simplified Bellmann update
    def policy_evaluation(self, P, U):
        U_next = [row[:] for row in U]
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if self.board[r][c] == 'X':
                    continue
                U_next[r][c] = self.Q_value([r, c], P[r][c], U)
        return U_next
                
    # Method to implement policy iteration that returns the optimal policy for a given MDP
    def modified_policy_iteration(self):
        U = [[0.0] * self.size[1] for i in range(self.size[0])]
        P = [[random.randint(0, len(self.actions)) for i in range(self.size[1])] for j in range(self.size[0])]
        iteration = 0
        
        while True:
            U = self.policy_evaluation(P, U)
            unchanged = True
            
            for r in range(self.size[0]):
                for c in range(self.size[1]):
                    if self.board[r][c] == 'X':
                        continue
                    max_val = float('-inf')
                    opt_action = 0
                    for i in range(len(self.actions)):
                        val = self.Q_value([r, c], i, U)
                        if val > max_val:
                            max_val = val
                            opt_action = i
                    
                    if max_val > U[r][c]:
                        P[r][c] = opt_action
                        unchanged = False

            if unchanged:
                break
            iteration += 1

        return self.get_policy(P)


# Function to parse all required parameters required to create a mdp from the input dictionary
def parse_and_create_mdp(input_dict):
    size = [int(s) for s in input_dict['size'].split(' ')]
    walls = []
    for wall in input_dict['walls'].split(','):
        walls.append([int(w) - 1 for w in wall.strip().split(' ')])
    terminal_states = []
    for state in input_dict['terminal_states'].split(','):
        s = state.strip().split(' ')
        terminal_states.append([int(s[0]) - 1, int(s[1]) - 1, float(s[2])])
    reward = float(input_dict['reward'])
    probs = [float(p) for p in input_dict['transition_probabilities'].strip().split(' ')]
    discount = float(input_dict['discount_rate'])
    epsilon = float(input_dict['epsilon'])
    
    return mdp(size, walls, terminal_states, reward, probs, discount, epsilon)


def main():
    with open('mdp_input.txt', 'r') as f:
        input_content = f.read()
    
    input_content = input_content.split('\n')
    
    input_dict = {}
    for line in input_content:
        if '#' not in line and line != '':
            input_dict[line.split(':')[0].strip()] = line.split(':')[1].strip()

    mdp = parse_and_create_mdp(input_dict)
    
    global out_string
        
    out_string += ('Size: ' + str(mdp.size) + '\n\n')
    out_string += ('Probabilities: ' + str(mdp.probs) + '\n\n')
    out_string += ('Discount_rate: ' + str(mdp.discount) + '\n\n')
    out_string += ('Epsilon: ' + str(mdp.epsilon) + '\n\n')
    out_string += ('Board:\n' + '\n'.join(' '.join(map(str, row)) for row in mdp.board) + '\n\n')
    
    out_string += ('--------------------------VALUE ITERATION----------------------\n\n')
    
    U = mdp.value_iteration()
    P = mdp.get_optimal_policy(U)
    
    out_string += ('Optimal Policy for Value iteration:\n')
    out_string += ('\n'.join(' '.join(map(str, row)) for row in P) + '\n\n')
    
    out_string += ('----------------------MODIFIED POLICY ITERATION--------------\n\n')
    
    P = mdp.modified_policy_iteration()
    out_string += ('Optimal Policy for Modified Policy iteration:\n')
    out_string += ('\n'.join(' '.join(map(str, row)) for row in P) + '\n\n')

    with open('mdp_output.txt', 'w') as f:
        f.write(out_string)

if __name__ == '__main__':
    main()