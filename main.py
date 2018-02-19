import gzip, pickle, numpy as np, time; from scipy.special import expit
with gzip.open('mnist.pkl.gz','rb') as ff :
	u = pickle._Unpickler( ff )
	u.encoding = 'latin1'
	train, val, test = u.load()

class Network(object):
	def __init__(self):
		#Constants
		self.INPUTLAYERSIZE = 785 #28*28+1for the bias
		self.OUTPUTLAYERSIZE = 10
		self.weights = np.random.uniform(-1,1,size=(self.INPUTLAYERSIZE,self.OUTPUTLAYERSIZE))

		#Hyperparameters to be tuned
		self.LEARNINGRATE = .1
		self.BATCHSIZE = 100

	def forward(self, x):
		return np.matmul(x,self.weights)

	def calcYhat(self, raw):
		#Get the index of the max value of each row
		guessIndex = np.argmax(raw,axis=1)
		#Encode the guesses and labels as 0's and 1
		return self.oneHot(guessIndex)

	def calcGradient(self, x, y, yhat):
		#xT(y-yhat)/-5
		return np.matmul(np.transpose(x),(y-yhat)/-5)

	def updateWeights(self,gradient):
		self.weights -= gradient

	def oneHot(self,data):
		#10000,
		zeros = np.zeros((data.shape[0],self.OUTPUTLAYERSIZE))
		#print(zeros.shape)
		zeros[np.arange(data.shape[0]),data] = 1
		return zeros

	def accuracy(self, y, yhat):
		correct = y == yhat
		acc = np.mean(np.alltrue(correct, axis=1))
		return acc

	def calcLoss(self, y, yhat):
		sqrErr=(y-yhat)**2
		meanSqrErr = np.mean(sqrErr)
		return meanSqrErr



if __name__ == "__main__":
	np.set_printoptions(linewidth=175)#Print final numbers on one line
	#np.get_printoptions()

	numEpoch = 10000
	nn = Network()

	#50000x785
	#added the 1's onto the end
	trainData = np.concatenate([train[0],np.ones((train[0].shape[0],1))],axis=1)
	#50000x1
	trainLabels = train[1]

	validateData = np.concatenate([val[0],np.ones((val[0].shape[0],1))],axis=1)
	validateLabels = val[1]

	loss = np.Inf

	for j in range(0,numEpoch):
		print("Epoch number", j)
		start = time.time()#Just for timing
		for i in range(0,trainData.shape[0],nn.BATCHSIZE):
		#for i in range(0,0+nn.BATCHSIZE,nn.BATCHSIZE):
			#create matricies of training data and labels
			data = trainData[i:i+nn.BATCHSIZE]
			labels = trainLabels[i:i+nn.BATCHSIZE]
			#BATCHSIZEx10 estimates
			rawOutput = nn.forward(data)
			yhat = nn.calcYhat(rawOutput)
			y = nn.oneHot(labels)
			gradient = nn.calcGradient(data, y, yhat)
			nn.updateWeights(gradient)

		end = time.time()
		print("\tTime for Epoch:",end-start)

		rawOutput = nn.forward(validateData)
		yhat = nn.calcYhat(rawOutput)
		y = nn.oneHot(validateLabels)

		acc = nn.accuracy(y,yhat)*100
		newLoss = nn.calcLoss(y,yhat)
		print("\tAccuracy: %f"%(acc))
		print("\tLoss: %f"%(newLoss))
		if((newLoss-loss)<.003):
			loss = newLoss
		else:
			break








