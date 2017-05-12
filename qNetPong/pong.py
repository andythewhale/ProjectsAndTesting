"""Andrew J Miller
5/11/2017"""
#The first thing we have to do is build pong from scratch.

#So first we're going to import the pygame library.
#Pygame allows us to easily create games in our env.
import pygame
import random #helps us define direction

#Defining variables:
frameRate = 49
windowHeight = 590
windowWidth = 590

#This is kind of weird but for pygame we need to define the stick params.
stickThickness = 13
stickLength = 69
stickBuffer = 13

#Also we need to define the ball params
ballWidth = 7
ballHeight = 7

#Also the ball speed and stick speed
yBallSpeed = 3
xBallSpeed = 5
stickSpeed = 5

#Make my game pretty again:
white = (255,255,255)
black = (0,0,0)
green = (85,107,47)
red = (255,0,0)

#Finally lets start initializing the game:
screen = pygame.display.set_mode(windowWidth, windowHeight)

#Function to draw our ball:
def drawBall(ballX, ballY):
	ball = pygame.rect(ballX, ballY, ballWidth, ballHeight)
	pygame.draw.rect(screen, green, ball)

#Function to draw our sticks:
#We'll make stickOne be the learner
def drawStickOne(stickOneY):
	stickOne = pygame.rect(stickBuffer, stickOneY, stickThickness, stickLength)	
	pygame.draw.rect(screen, white, stickOne)

#Now for the opposition paddle
def drawstickDos(stickDosY):
	stickDos = pygame.rect(windowWidth - stickBuffer - stickThickness, stickDosY, stickThickness, stickLength)
	pgame.draw.rect(screen, red, stickDos)

#Okay so we need a function where the ball gets moved.
#The ball needs to move or there's no game right?
def moveBall(stickOneY, stickDosY, ballX, ballY, ballXDir, ballYDir):
	ballX = ballX + ballXDir + ballXSpeed
	ballY = ballY + ballYDir + ballYSpeed
	score = 0

	#we need to check for collisions to interact when there is a move. This is our learning side.
	if (ballX <= stickBuffer + stickThickness and ballY + ballHeight >= stickOneY and ballY - ballHeight <= stickOneY + stickLength)
		ballXDir = 1
	elif (ballX <= 0):
		ballXDir = 1
		score = -1

		return [score, stickOneY, stickDosY, ballX, ballY, ballXDir, ballYDir]

	#We also need to check if it hits the opposing side.
	if (ballX >= windowWidth - stickThickness - stickBuffer and ballY + ballHeight >= stickDosY and ballY - ballHeight <= stickDosY + stickLength):
		ballXDir = -1
	elif (ballX >= windowWidth - ballWidth):
		ballXDir = -1
		score = 1

		return [score, stickOneY, stickDosY, ballX, ballY, ballXDir, ballYDir]

		#But what happens if it hits the top/bottom?:
		#Top:
		if (ballY <= 0):
			ballY = 0
			ballYDir = 1

		elif (ballY >= windowHeight - ballHeight):
			ballY = windowHeight - ballHeight
			ballYDir = 1

		return [score, stickOneY, stickDosY, ballX, ballY, ballXDir, ballYDir]

#We also need to move our paddles and check the position of our paddles.
def moveStickOne(action, stickOneY):
	#If statement for moving up
	if (action[1] == 1):
		stickOneY = stickOneY - stickSpeed
	#If statment for moving down
	if (action[2] == 1):
		stickOneY = stickOneY + stickSpeed

	#Can't move off the screen though, that's bad.
	if (stickOneY < 0):
		stickOneY = 0
	if (stickOneY > windowHeight - stickLength):
		stickOneY = windowHeight - stickLength
	return stickOneY

#PaddleDos needs some special rules because it is not learning. 
#We need to hardcode this player.
#Same stuff as before but minus the action
def moveStickDos(stickDosY, ballY):
	if (stickDosY + stickLength/2 < ballY + ballHeight/2):
		stickDosY = stickDosY + stickSpeed
	if (stickDosY + stickLength/2 > ballY + ballHeight/2):
		stickDosY = stickDosY - stickSpeed
	if (stickDosY < 0):
		stickDosY = 0
	if (stickDosY > windowHeight - stickLength):
		stickDosY = windowHeight - stickLength
		retrun stickDosY

#Let's make that game
class __init__(self):
	#Random initial direction for the ball:
	num = random.randint(0,9):
	#Keeping the score saved:
	self.tally = 0

	#Give initial positions to our sticks:
	self.stickOneY = windowHeight / 2 - stickLength / 2
	self.stickDosY = windowHeight / 2 - stickLength / 2

	#Ball direction
	self.ballYDir = 1
	self.ballXDir = 1

	#Start here:
	self.ballX = windowWidth / 2 - ballWidth / 2

	#Random ball movement:
	if (0 < num < 4):
		self.ballXDir = 1
		self.ballYDir = 1
	if (4 <= num < 5):
		self.ballXDir = -1
		self.ballYDir = 1
	if (5 <= num < 7):
		self.ballXDir = 1
		self.ballYDir = -1
	if (7 <= num < 10):
		self.ballXDir = -1
		self.ballYDir = -1
	num = random.randint(0,9)
	self.ballY = num*(windowHeight - ballHeight) / 9

#We need to see our current space in accordance with out frame rate:
def getCurrentFrame(self):
	pygame.event.pump()
	screen.fill(black)
	drawStickOne(self.stickOneY)
	drawstickDos(self.stickDosY)
	drawBall(self.ballX, self.ballY)

	#We have to have a way to read the image data
	image_data = pygame.surfarray.array3d(pygame.display.get_surface())
	pygame.display.flip()
	return image_data

#We need to update our screen in accordance with our frame rate:
def getNewFrame(self, action):
	pygame.event.pump()
	score = 0
	screen.fill(black)

	#Stick movement updates:
	#Learner:
	self.stickOneY = moveStickOne(action, self.stickOneY)
	drawStickOne(self.stickOneY)

	#Game AI:
	self. stickDosY = moveStickDos(self.stickDosY, self.ballY)
	drawstickDos(self.stickDosY)
	[score, self.stickOneY, self.stickDosY, self.ballX, self.ballY, self.ballXDir, self.ballYDir] = moveBall(self.stickOneY, self.stickDosY, self.ballX, self.ballY, self.ballXDir, self.ballYDir)
	drawBall(self.ballX, self.ballY)

	#Getting and displaying image data again.
	image_data = pygame.surfarray.array3d(pygame.display.get_surface())
	pygame.display.flip()

	#We have to update the score yo
	self.tally = self.tally+score
	print("Tally is" + str(self.tally))
	return[score, image_data]