"""
Andrew J Miller
5/11/17
-The idea here is to read pixel data and create 2 Pong AI's
-They will play against eachother
-We will use tensorflow
-We will be using Deep Mind's approach utilizing Deep Q
-Deep Q is the idea that we only feed the machine a fitness score, pixels,
...controls, etc, and the machine figures out how to play.
-We will be utilizing re-enforcement learning (duh score)
"""

#This is the Re-enforcement algorithm for the pong game!
#The actual game is in the pong.py file.

#Imports:
import tensorflow as tf
import cv2 
import pong
import numpy as np
import random
from collections import deque

#We need our abilities and our learning rate:
actions = 3 #Up, Down, Still
learningRate = 0.9
#We want our learning rate to change as it gets closer to a maxim
initialLRG = 1
finalLRG = 0.013
#We need a number of observation/exploration points to change the LR:
observation = 49000
exploration = 490000
#We need a certain amount of memory:
memory = 490000
#We will train on sets of 113
batch = 113

#Tensorflow graph:
#Let's make our zero graph:
wConv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
bConv1 = tf.Variable(tf.zeros([32]))

wConv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
bConv2 = tf.Variable(tf.zeros([64]))

wConv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
bConv3 = tf.Variable(tf.zeros([64]))

wfc4 = tf.Variable(tf.zeros([3136, 784]))
bfc4 = tf.Variable(tf.zeros([784]))

wfc5 = tf.Variable(tf.zeros([784, actions]))
bfc5 = tf.Variable(tf.zeros([actions]))

#Rectified linear unit activation on a 2D convolution given a 4D input
conv1 = tf.nn.relu(tf.nn.conv2d(s, wConv1, strides = [1, 4, 4, 1], padding = 'VALID') + bConv1)

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, wConv2, strides = [1, 2, 2, 1], padding = 'VALID') + bConv2)

conv3 = tf.nn.relu(tf.nn.conv2d(conv2 ,wConv3, strides = [1, 1, 1, 1], padding = 'VALID') + bConv3)

conv3Flattened = tf.reshape(conv3, [-1, 3136])

fc4 = tf.nn.relu(tf.matmul(conv3Flattened, wfc4) + bfc4)

fc5 = tf.matmul(wfc4, wfc5) + bfc5

return (s, fc5)

#This is the deepQ network:
def trainGraph(inp, out, sess):

	#Multiply predicted output by a 1 value vector and the rest set to 0
	argmax = tf.placeholder('float', [None, actions])
	gt = tf.placeholder('float', [None])

	action = tf.reduce_sum(tf.multiply(out,argmax), reduction_indices = 1)
	
	#Minimize cost through backpropogation
	cost = tf.reduce_mean(tf.square(action-gt))
	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

	game = pong.game()

	d = deque()

	#Initial frame
	frame = frame.getCurrentFrame()
	#convert rgb to gray scale for processing
    #frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    #binary colors, black or white
    #ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)

    #stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    #Saver
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    e = initialLRG

    #Training loop:
    while(1):
    	#output
    	out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
    	#argmax
    	argmax_t = np.zeros([actions])

    	if(random.random() <= e):
    		maxIndex = random.randrange(actions)
    	else:
    		maxIndex = np.argmax(out_t)
    	argmax_t[maxIndex] = 1

    	if (e > finalLRG):
    		e -= (initialLRG - finalLRG) / exploration

    	#Reward circuit
    	reward_t, frame = game.getNewFrame(argmax_t)
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))

        #New input bae
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        #save it
        d.append((inp_t, argmax_t, reward_t, inp_t1))

        #If we run out of memory make room:
        if (len(d) > memory):
        	D.popleft()

        #Training iteration
        if (t > observation):

        	minibatch = random.sample(d, batch)

        	inp_batch = [d[0] for f in minibatch]
        	argmax_batch = [d[1] for d in minibatch]
        	reward_batch = [d[2] for d in minibatch]
        	inp_t1_batch = [d[3] for d in minibatch]

        	gt_batch = []
        	out_batch = out.eval(feed_dict = {inp : inp_t1_batch})

        	#We need values and morals:
        	for i in range(0, len(minibatch)):
        		gt_batch.append(reward_batch[i] + learningRate * np.max(out_batch[i]))

        	#train on it	
        	train_step.run(feed_dict = {gt: gt_batch, argmax: argmax_batch, inp: inp_batch})
        	inp_t = inp_t1
        	t = t + 1

        	if (t % 10000 == 0):
        		saver.sess(sess, './' + 'pong' + '-dqn', global_step = t)

        	print('timestep', t, '..epsilon', e, '.. action', maxIndex, '..reward', reward_t, '...qMax', np.max(out_t))



def main():
	#create session
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = createGraph()
    #train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()





