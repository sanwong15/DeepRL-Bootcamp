import numpy as np
# import cPickle as pickle
import pickle as pk # For Python 3
import gym
import gym.spaces

'''
Vanilla Policy Gradient
Initialize policy parameter THETA, Baseline b

for iteration = 1,2 ... do
	Collect a set of Trajectories (i.e. Path) by executing the current policy

	AT EACH TIME STAMP in EACH Trajectories, Compute and Return (1) R_t = SUM OVER t to T-1 (Alpha^(t'-t) * r_t')
	(2) Advantage Estimate: A_head_t = R_t - b(s_t)

	Re-fit the baseline: By minimizing Squrt 2 norm of (b(s_t) - R_t)

	Sum Over all trajectories and time steps

	Update the policy using a Policy Grad Estimate g_head
'''




# Set up Hyperparameters
H = 200 # Number of Hidden layer neurons
batch_size = 10 #  How many episodes to do a param update
learning_rate = 1e-4
gamma = 0.99 # Discount factor for reward
decay_rate = 0.99 # Decay factor for RMSProp leaky sum of grad^2
resume = False
render = False


# Model initialization
D = 80*80 # Define grid size (i.e: ScreenShot size of Pong Game)

if resume:
	model = pickle.load(open('save.p', 'rb'))
else:
	model = {}
	model['W1'] = np.random.randn(H,D)/np.sqrt(D)
	model['W2'] = np.random.randn(H)/np.sqrt(H)

# np.zero_like return an array of zeros with the same shape and type as a given array
# grad_buffer = {k:np.zeros_like(v) for k,v in model.iteritems()} # This only works with Python2
# rmsprop_cache = {k:np.zeros_like(v) for k,v in model.iteritems()}

# In order to work with Python 3. Made the following changes
grad_buffer = {k: np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k:np.zeros_like(v) for k,v in model.items()}

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def prepro(I):
	'''
	pre-process 210*160*3 uint8 frame into 6400 (80*80) 1d float vector
	'''

	# Crop the screen
	I = I[35:195]
	I = I[::2,::2,0] # Downsample by the factor of 2
	I[I==144] = 0 # Erase background type 1
	I[I==109] = 0 # Erase background type 2
	I[I != 0] = 1 # Everything else (paddles, ball) are set to 1
	return I.astype(np.float).ravel()

def discount_rewards(r):
	'''
	Input: 1d Float array of rewards (reward with the Horizon)
	Output: Discounted reward (array, same size as r)
	'''
	discounted_r = np.zeros_like(r)
	running_add = 0

	# The discounted_r shows the reward (also consider the discounted one in future time) at diff time step
	# for t in reversed(xrange(0,r.size)): # For Python 2
	for t in reversed(range(0,r.size)): # For Python 3
		if r[t] != 0:
			running_add = 0 # reset the sum. Game boundary

		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def policy_forward(x):
	h = np.dot(model['W1'], x)
	h[h<0] = 0 # ReLU
	logp = np.dot(model['W2'], h)
	p = sigmoid(logp)

	return p, h

def policy_backward(eph, epdlogp):
	'''
	Backward Pass:
	eph is array of intermediate hidden state
	epdlogp is the log_prob of (1) Going Up and (2) Going Down
	'''
	dW2 = np.dot(eph.T, epdlogp).ravel() # ravel returns contiguous array (1-D array with all teh input array elements and with the same type as it)
	dh = np.outer(epdlogp, model['W2']) # compute outer product
	'''
	>>> a = np.array([[1, 2], [1, 1]])
	array([[1, 2],
    	 [1, 1]])
	>>> b = np.array([[2, 3], [1, 3]])
	array([[2, 3],
    	 [1, 3]])
	>>> np.outer(a, b)
	array([[2, 3, 1, 3],
    	 [4, 6, 2, 6],
    	 [2, 3, 1, 3],
    	 [2, 3, 1, 3]])
	'''
	dh[eph <= 0] = 0
	dW1 = np.dot(dh.T, epx)
	return {'W1':dW1, 'W2':dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # Will use this to compute frame difference
xs, hs, dlogps, drs = [],[],[],[]
# xs book-keep the screen input: x for each time step
# hs book-keep the hidden layer value at each time step (where a new screenshoot is passed into the network and perform a feed forward)
# I believe the "d" below actually stands for diff
# dlogps book-keep the difference between the Fake label and our log_prob calculated by the network
# drs book-keep the reward for each action taken at each time step
running_reward = None
reward_sum = 0
episode_number = 0

while True:
	if render: env.render()

	# Current input: x
	cur_x = prepro(observation)
	# Calculate the frame difference
	x = cur_x - prev_x if prev_x is not None else np.zeros(D)
	# Update the prev_x for next time use
	prev_x = cur_x

	# Feed Forward
	aprob, h = policy_forward(x)
	action = 2 if np.random.uniform() < aprob else 3

	# Record various intermediates (this will be needed for backprop)
	xs.append(x)
	hs.append(h)

	# Create Fake label
	y = 1 if action == 2 else 0
	dlogps.append(y-aprob) # dlogps store the difference between y and the feedforward aprob

	# Take the action and get the new observation and measurements
	observation, reward, done, info = env.step(action) # The "reward" here is the resultant reward of taking one action at one time-step
	reward_sum += reward # The reward result in taking the action will then be considered in a accumlative manner. And hence the reward_sum

	drs.append(reward)  # Not yet discounted yet. Just record how much reward one can get

    # When it is end game
	if done:
		episode_number += 1

		# Stack together all inputs, hidden states, action gradients and rewards for ths episode

		# epx: book keep all the screenshoot (x) happened in this episode
		# eph: book keep all the hiddle value changes regards to the feed forward network (due to diff screen shoot)
		# epdlogp: book keep all the difference between Fake label: y and calcualted result: logp in this episode
		# epr: book keep all the cumanlative reward at all time step within this episode
		epx = np.vstack(xs)
		eph = np.vstack(hs)
		epdlogp = np.vstack(dlogps)
		epr = np.vstack(drs)
		# Reset the array
		xs, hs, dlogps, drs = [],[],[],[]

		# Compute the Discounted reward backwards through time
		discounted_epr = discount_rewards(epr)
		# Standardize the rewards to be unit normal
		discounted_epr -= np.mean(discounted_epr)
		discounted_epr /= np.std(discounted_epr)

		epdlogp *= discounted_epr # The A_i?
		grad = policy_backward(eph, epdlogp)

		for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

		# Perform RMSProp update very batch_size episodes
		if episode_number % batch_size == 0:
			#for k, v in model.iteritems(): # Python2
			for k, v in model.items(): # Python3
				g = grad_buffer[k]
				rmsprop_cache[k] = decay_rate*rmsprop_cache[k] + (1-decay_rate)*g**2
				model[k] += learning_rate*g/(np.sqrt(rmsprop_cache[k] + 1e-5))
				grad_buffer[k] = np.zeros_like(v) # Reset grad buffer

		running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
		print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
		if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

		reward_sum = 0
		observation = env.reset() # reset env
		prev_x = None


	if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
		# print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')	
		print ('ep {}'.format(episode_number) + ': game finished, reward: {}'.format(reward)  + ('' if reward == -1 else ' !!!!!!!!'))
		#print("reward != 0")







