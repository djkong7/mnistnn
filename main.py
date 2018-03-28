import os, sys, time, numpy as np, idx_parser as cparser
from scipy.special import expit

class Network(object):
	def __init__(self):
		#Constants
		self.INPUTLAYERSIZE = 785 #28*28+1for the bias
		self.OUTPUTLAYERSIZE = 10
		self.weights = np.random.uniform(-1,1,size=(self.INPUTLAYERSIZE,self.OUTPUTLAYERSIZE))

		#Hyperparameters to be tuned
		self.LEARNINGRATE = .05
		self.BATCHSIZE = 100

	def forward(self, x):
		z = np.matmul(x,self.weights)
		return self.sigmoid(z)

	def sigmoid(self,x):
		return expit(x)

	def calcYhat(self, output):
		return self.oneHot(np.argmax(output, axis=1))

	def calcGradient(self, x, y, output):
		#(xT)DOT((y-output)/-5)*sig(z)*(1-sig(z))
		#z=xW
		z = np.matmul(x,self.weights)
		a = self.sigmoid(z)
		#sig(z)*(1-sig(z))
		sigGrad = np.multiply(a,1-a)
		#((y-output)/-5)*sig(z)*(1-sig(z))
		outputGrad = np.multiply((y-output)/-5,sigGrad)
		#(xT)DOT((y-output)/-5)*sig(z)*(1-sig(z))
		gradient = np.matmul(np.transpose(x),outputGrad)
		#print("\n",gradient,"\n")
		return gradient*self.LEARNINGRATE

	def updateWeights(self, gradient):
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

	def calcLoss(self, y, output):
		sqrErr=(y-output)**2
		meanSqrErr = np.mean(sqrErr)
		return meanSqrErr



def train(nn, train_images, train_labels, validate_images, validate_labels):
	num_epoch = 1000
	stop = False
 
	loss = np.Inf
	counter = 0
	loss_increase = 0

	error = []
	accuracy = []
	losses = []

	for j in range(0,num_epoch):
		print("Epoch number", j)
		start = time.time()#Just for timing
		for i in range(0,train_images.shape[0],nn.BATCHSIZE):
		#for i in range(0,1,nn.BATCHSIZE):
			#create matricies of training data and labels
			data = train_images[i:i+nn.BATCHSIZE]
			labels = train_labels[i:i+nn.BATCHSIZE]
			#BATCHSIZEx10 estimates
			#z=preactivation
			a1 = nn.forward(data)
			y = nn.oneHot(labels)
			#Calc error for logging
			#Append for a single write at the end
			error.append(nn.calcLoss(y,a1))

			gradient = nn.calcGradient(data, y, a1)
			nn.updateWeights(gradient)

			counter += 1

			if (counter % 150) == 0:
				a1 = nn.forward(validate_images)
				yhat = nn.calcYhat(a1)
				y = nn.oneHot(validate_labels)

				acc = nn.accuracy(y,yhat)*100
				newLoss = nn.calcLoss(y,a1)
				accuracy.append(acc)
				losses.append(newLoss)

				#Automatically stop to prevent overtraining.
				if (newLoss - loss) <= 0:
					loss = newLoss
					loss_increase = 0
					save_weights = nn.weights
					#print("\tAccuracy: %f"%(acc))
					#print("\tLoss: %f"%(newLoss))
				else:
					loss_increase += 1
					if loss_increase >= 8:
						#Stop epoch loop
						stop = True
						#Stop batchsize loop
						break

		end = time.time()
		print("\tTime for Epoch:",end-start)
		if stop:
			nn.weights = save_weights
			break

def test(nn, test_images, test_labels, wmu_test_images, wmu_test_labels):
	a1 = nn.forward(test_images)
	yhat = nn.calcYhat(a1)
	y = nn.oneHot(test_labels)

	acc = nn.accuracy(y,yhat)*100
	newLoss = nn.calcLoss(y,a1)
	print("MNIST")
	print("\tAccuracy: %f"%(acc))
	print("\tLoss: %f"%(newLoss))

	a1 = nn.forward(wmu_test_images)
	yhat = nn.calcYhat(a1)
	y = nn.oneHot(wmu_test_labels)

	acc = nn.accuracy(y,yhat)*100
	newLoss = nn.calcLoss(y,a1)
	print("WMU")
	print("\tAccuracy: %f"%(acc))
	print("\tLoss: %f"%(newLoss))

def load_data():
	try:
		print("Loading numpy data.")
		#Numpy files are much faster to load than the mnist data
		#Try to load these first
		train_images = np.load("data/nmpy/train_images.npy")
		train_labels = np.load("data/nmpy/train_labels.npy")
		test_images = np.load("data/nmpy/test_images.npy")
		test_labels = np.load("data/nmpy/test_labels.npy")
		wmu_test_images = np.load("data/nmpy/wmu_test_images.npy")
		wmu_test_labels = np.load("data/nmpy/wmu_test_labels.npy")

	except IOError as e1:
		print(e1)
		print("Numpy hasn't been written. Trying idx...")

		try:
			#Load the data from mnist
			train_images = cparser.idx("data/mnist/train_images.idx")
			train_labels = cparser.idx("data/mnist/train_labels.idx")
			test_images = cparser.idx("data/mnist/test_images.idx")
			test_labels = cparser.idx("data/mnist/test_labels.idx")
			wmu_test_images = cparser.idx("data/wmu/wmu_test_images.idx")
			wmu_test_labels = cparser.idx("data/wmu/wmu_test_labels.idx")

			#images come back as a 3d np array.
			#reshpae the image data, normalize it(retypes to floats), and append 1's
			shape = train_images.shape
			train_images = (train_images.reshape(shape[0], 28*28))/255
			train_images = np.concatenate([train_images,np.ones((shape[0],1))],axis=1)
			
			shape = test_images.shape
			test_images = (test_images.reshape(shape[0], 28*28))/255
			test_images = np.concatenate([test_images,np.ones((shape[0],1))],axis=1)

			shape = wmu_test_images.shape
			wmu_test_images = (wmu_test_images.reshape(shape[0], 28*28))/255
			wmu_test_images = np.concatenate([wmu_test_images,np.ones((shape[0],1))],axis=1)

			#Make the mnpy directory if it doesn't exist.
			if not os.path.exists("data/nmpy"):
				os.makedirs("data/nmpy")

			np.save("data/nmpy/train_images", train_images)
			np.save("data/nmpy/train_labels", train_labels)
			np.save("data/nmpy/test_images", test_images)
			np.save("data/nmpy/test_labels", test_labels)
			np.save("data/nmpy/wmu_test_images", wmu_test_images)
			np.save("data/nmpy/wmu_test_labels", wmu_test_labels)

		except IOError as e2:
			print("Couldn't find file.", e2)
			print("Exiting...")
			sys.exit(1)
		
	#Create validation set
	validate_images = train_images[50000:]
	train_images = train_images[0:50000]
	
	validate_labels = train_labels[50000:]
	train_labels = train_labels[0:50000]

	return train_images, train_labels, validate_images, validate_labels, test_images, test_labels, wmu_test_images, wmu_test_labels
	

if __name__ == "__main__":
	np.set_printoptions(linewidth=175)#Print final numbers on one line
	#np.set_printoptions(threshold=np.nan)#Print entire array
	#np.get_printoptions()
	
	nn = Network()
	
	train_images, train_labels, validate_images, validate_labels, test_images, test_labels, wmu_test_images, wmu_test_labels = load_data()
	train_start = time.time()
	train(nn, train_images, train_labels, validate_images, validate_labels)
	train_end = time.time()
	print("Total time to train:", train_end-train_start)
	test(nn, test_images, test_labels, wmu_test_images, wmu_test_labels)













