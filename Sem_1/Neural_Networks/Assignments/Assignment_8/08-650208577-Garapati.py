import numpy as np

# np.random.seed(2021)


def get_reward_and_next_state(x, a, n, G, coins):
	new_state = x
	reward = 0
	actions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
	x_idx = x[0] + actions[a][0]
	y_idx = x[1] + actions[a][1]
	if (x_idx >= 1 and x_idx <= n) and (y_idx >= 1 and y_idx <= n):
		new_state = (x_idx, y_idx)
	# print(new_state)
	if new_state[0] == n and new_state[1] == 1:
		# print('New State: {}, Coins: {}'.format(new_state, coins))
		if coins < G:
			coins += 1
		new_state = x
	elif new_state[0] == 1 and new_state[1] == n:
		# print('New State: {}, Reward: {}'.format(new_state, coins))
		reward += coins
		coins = 0
		new_state = x
	return reward, new_state, coins


def compute_actions_and_cumulative_reward(Q_table, n, G, gamma):
	coins = 0
	x = (1, 1)
	states = [(1, 1)]
	a = 0
	tot_reward = 0
	actions = []
	actions_taken = ['U', 'L', 'D', 'R']
	for t in range(150):
		a = np.argmax(Q_table[x[0] - 1, x[1] - 1])
		r, y, coins = get_reward_and_next_state(x, a, n, G, coins)
		tot_reward += np.power(gamma, t) * r
		actions.append(actions_taken[a])
		states.append(y)
		x = y
	print(tot_reward)
	print(actions)
	print(states)


# States = {(1, 1), (2, 1) ...}
# Actions = {U, L, D, R}
def Q_learning(n, G, gamma, e, alpha):
	states = n*n
	actions = 4
	actions_list = ['U', 'L', 'D', 'R']
	Q_table = np.random.normal(0, 1, (n, n, actions))
	# Q_table = np.zeros((n*n, actions))
	# print(Q_table.shape)
	episodes = 1

	for episode in range(episodes):
		print('Episode: {}'.format(episode))
		print(Q_table)
		x = (1, 1)
		coins = 0
		a = 0
		y = 0
		for t in range(10000):
			c = int(np.random.choice([1, 2], 1, p=[e, 1-e]))
			if c == 1:
				a = np.random.choice([0, 1, 2, 3])
			else:
				a = np.argmax(Q_table[x[0] - 1, x[1] - 1])
			# print(actions_list[a])
			r, y, coins = get_reward_and_next_state(x, a, n, G, coins)
			if r > 0:
				print('Current: {}, Action:{}, Next: {}, Coins: {}, Reward: {}'.format(x, actions_list[a], y, coins, r))
			# print(coins)
			# print(Q_table[(x[0] - 1)*n + (x[1] - 1)][a], np.max(Q_table[(y[0] - 1)*n + (y[1] - 1)]))
			Q_table[x[0] - 1, x[1] - 1, a] = (1 - alpha) * Q_table[x[0] - 1, x[1] - 1, a] + alpha*(r + gamma * np.max(Q_table[y[0] - 1, y[1] - 1]))
			x = y
		compute_actions_and_cumulative_reward(Q_table, n, G, gamma)
		# print(Q_table)
	return Q_table


if __name__ == '__main__':
	print(np.random.get_state()[1][0])
	n = 5
	G = 3
	gamma = 0.9
	e = 0.1
	alpha = 0.99

	Q_table = Q_learning(n, G, gamma, e, alpha)
	print(Q_table)
	compute_actions_and_cumulative_reward(Q_table, n, G, gamma)
