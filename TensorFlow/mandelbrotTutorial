#This is a a follow along to Google's tensorflow tutorial.
#This is a a follow along to Google's tensorflow tutorial.
#I'll follow along here and then upload an ipython notebook that debugs the program.

"""So a quick overview on this stuff:
		-We're going to be visualizing the mandelbrot set. 
		-The mandelbrot set is a set of complex numbers
		-Complex numbers are numbers that must be expressed in the format of z = a+bi
		-They have an imaginary unit and they're really weird. i^2 = -1
		-On the mandelbrot set the complex numbers do not diverge on the function...
		- fc(z) = z^2 + c when iterated from z = 0
"""

#Importing Libraries
import tensorflow as tf
import numpy as np

#Importing vvisualization
import PIL.Image
from io import BytesIO
from IPython.display import Image, display

def DisplayFractal(a, fmt ='jpeg'):
	#Display an array of iteration counts as a colorful picture of a fractal

	a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	img = np.concatenate([10+20*np.cos(a_cyclic),
						  30+50*np.sin(a_cyclic),
						  155-80*np.cos(a_cyclic)], 2)

	img[a==a.max()] = 0
	a = img
	a = np.uint8(np.clip(a, 0, 255))
	f = BytesIO()
	PIL.Image.fromarray(a).save(f,fmt)
	display(Image(data=f.getvalue()))

#An interactive session is good for this but a regular session would work as well.
sess = tf.InteractiveSession()

#We need to use numpy to create a 2D array of complex numbers for us
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y

#Define and initialize tensors:
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

#Tensorflow requires that we explicitly initialize variables before using them.
tf.global_variables_initializer().run()

#Specifying more of the computations:
#Compute new values of z: z^2 + x
zs_ = zs * zs + xs

#Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

#Operation to update the zs and the iteration count:
#Note: We keep computing zs after they diverge. This is apparently wasteful.
#There is a better way to do this.

step = tf.group(
	zs.assign(zs_),
	ns.assign_add(tf.cast(not_diverged, tf.float32))
	)
	
#Now we run it a few hundred times so that we can eventually look at it.
for i in range(500): step.run()

#Display it:
DisplayFractal(ns.eval())

#This is the final version. The code was debugged in the ipython notebook by Andy Miller
#This version of the code may only work on the machine for which it was designed for
