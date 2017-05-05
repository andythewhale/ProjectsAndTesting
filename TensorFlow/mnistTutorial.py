"""
This is a tutorial explaining tensorflow and its applications. Specifically, this tutorial will cover the analysis of the MNIST dataset.
Everything will be compiled in an ipython notebook afterwords to make sure that the code is debugged and works. Thanks Google. 
Andy Miller, 5/4/2017

Background:

MNIST is the handwritten number dataset. It's a bunch of images with the arabic numerals on them. 0-9. The dataset includes labels.
We are going to use tensorflow to create a model to correctly identify the numerals the dataset is labaled so we can use supervised learning
I think we could also easily use re-enforcement learning. We are going to learn about softmax regressions. We are going to create a model for
recognizing digits based on pixels. We are going to understand how to validate accuracy using tensorflow

"""

#First we need to import our data:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Ok so this data is 55,000 data points of 28x28 pixel arragements of numbers, we're going to flatten these matrices into vectors to lower the dimensionality.
#This doesn't really simplify or lower the dimensionality, it just puts our data into an easier to handle array.
#However it does matter a little bit as the more complex and accurate image analysis algorithms use matrices (or maybe just 2 arrays, not sure yet)
#We're also going to do this thing called "one-hot" vectorizing. We're going to set all of the spaces with filled in stuff to 1 and nothing to 0.

#We're going to use an evidence equation in the format of y=mx+b where b is the bias. m is the weights. 
#We're going to use the softmax equation on our evidence function. Our softmax function is going to distribute our evidence over the 10 possible outputs.
# softmax(x) = normalize(exp(x)) or softmax(x) = exp(x)/sum(exp(x))

import tensorflow as tf

#We need the placeholder for a 784 length array of our 55,000 data points.
x = tf.placeholder(tf.float32, [None, 784])

#Now we need to set up our weight and bias variables.
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#We are going to learn w and b. Their initial values are set up as arrays of zeros.

#This is why tensorflow is awesome. That's our softmax function
y = tf.nn.softmax(tf.matmul(x, w) + b)

#We're going to use the cross entropy function to define the loss of our model.
#Hy'(y) = - sum(yi' * log yi)
#y' is just the real labeled data. yi is our guess for this iteration

#Here's the placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

#Now we implement the cross entropy function:
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1])) This is numerically unstable. Too good for ya, ungreatful

#This is stable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Training. Training happens via backpropagation.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#This step, minimizes cross entropy

#Now we launch an interactive sess!
sess = tf.InteractiveSession()

#We then need to call out our variables and initialize our session
tf.global_variables_initializer().run()

#Now we actually pick the number of loops we want to do to run our neural network on the data set:
for _ in range(1313):
	batch_xs, batch_ys, = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
#For each loop you will recieve a batch of 100 random data points from the training set 
#The train step then feeds the 100 data points, replacing the placeholders in the next loop
#This allows our NN to be trained based off of what it just saw in this step (how nice!)
#For future reference, google wants me to know that this is called stochastic training.
#This is a very inexpensive way of doing things, the expensive way would be to use all the data and not just 100 points.

#Let us do as the romans, and evaluate our model:

#Let's find out how to predict the correct label first:
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#tf.equal checks if they're equal y_ is the actual correct, y is the prediction, argmax checks the last arg (index of highest entry), which is the label.

#Check our accuracy yo:
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict= {x: mnist.test.images, y_: mnist.test.labels}))


