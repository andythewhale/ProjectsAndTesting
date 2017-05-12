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

#Importations:
import tensorflow as tf
import cv2 
import pong
import numpy as np
import random
from collections import deque

