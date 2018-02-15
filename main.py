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
		self.LEARNINGRATE = .00001
		
		self.weights = np.random.uniform(-1,1,size=(self.INPUTLAYERSIZE,self.OUTPUTLAYERSIZE))

		#Map the numerical guess to an array where 1 is the guess
		self.numLookup = dict([
			(0, np.matrix([1,0,0,0,0,0,0,0,0,0])),
			(1, np.matrix([0,1,0,0,0,0,0,0,0,0])),
			(2, np.matrix([0,0,1,0,0,0,0,0,0,0])),
			(3, np.matrix([0,0,0,1,0,0,0,0,0,0])),
			(4, np.matrix([0,0,0,0,1,0,0,0,0,0])),
			(5, np.matrix([0,0,0,0,0,1,0,0,0,0])),
			(6, np.matrix([0,0,0,0,0,0,1,0,0,0])),
			(7, np.matrix([0,0,0,0,0,0,0,1,0,0])),
			(8, np.matrix([0,0,0,0,0,0,0,0,1,0])),
			(9, np.matrix([0,0,0,0,0,0,0,0,0,1]))
		])

	def forward(self, x):
		outputRaw = np.dot(x,self.weights)
		return outputRaw

	def calcErr(self, x, output, labels):
		#Get the index of the max value of each row
		guess = output.argmax(1)

		#print("Output:\n", output)
		#print("Guess:\n", guess)
		answers = self.labelsToMatrix(labels)
		guesses = self.labelsToMatrix(guess)
		
		#Calculate the BATCHSIZEx10 err vector
		errVector = (answers - guesses)#/BATCHSIZE
		#print("Error Vector:\n", errVector)
		#Calulate the MSE
		#(y-yhat)^2
		#sumErrSquared = np.dot(errVector,errVector)
		#Take the mean
		#meanSquaredErr = sumErrSquared/errVector.shape[1]

		#Compute Gradient
		errVector *= -1/5
		#Make raw data 785xBATCHSIZE
		x = np.matrix.transpose(x)
		#Calculate 785x10 update matrix to add to weights matrix
		updateMatrix = np.dot(x,errVector)

		#Implement update
		self.weights-=updateMatrix*(self.LEARNINGRATE/BATCHSIZE)

	def labelsToMatrix(self, labels):
		answers = self.numLookup[labels.item(0,0)]
		for i in range(1,labels.shape[0]):
			answers = np.concatenate((answers, self.numLookup[labels.item(0,0)]), axis=0)
		#print("Label:\n",labels)
		#print("Answers:\n",answers)
		return answers


def dataToMatrix(array):
	#Append 1 for the bias calculations
	#Force the data into a matrix
	return np.mat(np.append(array, [1]))



if __name__ == "__main__":
	BATCHSIZE = 100
	np.set_printoptions(linewidth=175)#Print final numbers on one line
	#np.get_printoptions()

	numEpoch = 15
	nn = Network()

	for j in range(0,numEpoch):
		print("Epoch number", j)
		start = time.time()#Just for timing
		for i in range(0,train[0].shape[0],BATCHSIZE):
			#create matricies of training data
			dataMatrix = dataToMatrix(train[0][i])#initialize the matrix to append to
			labelMatrix = np.mat(train[1][i])#initialize the matrix to append to
			for k in range(i+1,i+BATCHSIZE): 
				#create a BATCHSIZEx785 matrix of training data
				dataMatrix = np.concatenate((dataMatrix,dataToMatrix(train[0][k])), axis=0)
				#create a BATCHSIZEx1 matrix of answers to the training data
				labelMatrix = np.concatenate((labelMatrix, np.mat(train[1][k])), axis=0)
			
			#BATCHSIZE by 10 estimates
			estimate = nn.forward(dataMatrix)
			nn.calcErr(dataMatrix, estimate, labelMatrix)
			#break

		end = time.time()
		print("\tTime for Epoch:",end-start)

		numWrong = np.array([0,0,0,0,0,0,0,0,0,0])
		for k in range(0,val[0].shape[0]):
			data = val[0][k]
			data = np.append(data, [1])
			data = np.mat(data)
			#print(a.shape)
			
			estimate = nn.forward(data)
			#print(estimate.shape, estimate)

			guess = np.argmax(estimate)

			if(guess != val[1][k]):
				numWrong[val[1][k]]+=1

		#print("\t",numWrong)
		#print("\t",(np.divide(numWrong,np.sum(numWrong)))*100)
		print("\tPercent wrong:",(np.sum(numWrong)/val[0].shape[0])*100,"%")
		