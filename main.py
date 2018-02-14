import gzip, pickle, numpy as np, time
with gzip.open('mnist.pkl.gz','rb') as ff :
	u = pickle._Unpickler( ff )
	u.encoding = 'latin1'
	train, val, test = u.load()

class Network(object):
	def __init__(self):
		#Constants
		self.inputLayerSize = 785 #28*28+1for the bias
		self.outputLayerSize = 10
		
		self.weights = np.random.uniform(-1,1,size=(self.inputLayerSize,self.outputLayerSize))

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

		self.learningRate = .5

	def forward(self, x):
		outputRaw = np.dot(x,self.weights)
		return outputRaw

	def calcErr(self, x, output, answer):

		
		#Get the index of the max value
		guess = np.argmax(output)

		#print("Guees: %d Answer: %d" %(guess,answer))
		#Calculate the 1x10 err vector
		errVector = (self.numLookup[answer] - self.numLookup[guess]) * self.learningRate
		x = np.matrix.transpose(x)
		#Calculate 785x10 err matrix to add to weights matrix
		errMatrix = np.dot(x,errVector)

		#Implement error
		self.weights+=errMatrix


if __name__ == "__main__":
	np.set_printoptions(linewidth=175)

	#np.get_printoptions()

	numEpoch = 3

	nn = Network()

	for j in range(0,numEpoch):
		print("Epoch number", j)
		start = time.time()
		for i in range(0,train[0].shape[0]):
		#for i in range(0,10):
			data = train[0][i]
			data = np.append(data, [1])
			data = np.mat(data)
			#print(a.shape)
			
			estimate = nn.forward(data)
			#print(estimate.shape, estimate)

			nn.calcErr(data, estimate, train[1][i])
		end = time.time()
		print("\tTime for Epoch:",end-start)

		numWrong = np.array([0,0,0,0,0,0,0,0,0,0])
		for k in range(0,test[0].shape[0]):
			data = test[0][k]
			data = np.append(data, [1])
			data = np.mat(data)
			#print(a.shape)
			
			estimate = nn.forward(data)
			#print(estimate.shape, estimate)

			guess = np.argmax(estimate)

			if(guess != test[1][k]):
				numWrong[guess]+=1

		print("\t",numWrong)
		print("\t",(np.divide(numWrong,np.sum(numWrong)))*100)
		print("\tPercent wrong:",(np.sum(numWrong)/test[0].shape[0])*100,"%")
		