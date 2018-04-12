from __future__ import print_function
import os, sys, time, torch as pt, numpy as np, idx_parser as cparser

LEARNING_RATE = .001
INPUT_SIZE = 784
NUM_CLASSES = 10
HIDDEN_LAYER_SIZE = 200

class Net(pt.nn.Module):
    def __init__(self, INPUT_SIZE, NUM_CLASSES):
        super(Net, self).__init__()
        self.input = pt.nn.Linear(INPUT_SIZE, HIDDEN_LAYER_SIZE)
        self.hidden1 = pt.nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.ouput = pt.nn.Linear(HIDDEN_LAYER_SIZE, NUM_CLASSES)
        
    def forward(self, x):
        x = 
        
        

def train(nn, train_images, train_labels, validate_images, validate_labels):
    num_epoch = 1000
    stop = False
 
    loss = np.Inf
    counter = 0
    loss_increase = 0

    error = []
    accuracy = []
    losses = []

    #For calculating loss
    criterion = pt.nn.modules.loss.CrossEntropyLoss()
    optimizer = pt.opim.SGD(net.parameters(), lr=LEARNING_RATE)
    
    for j in range(0,num_epoch):
        print("Epoch number", j)
        start = time.time()#Just for timing
        for i in range(0,train_images.shape[0],nn.BATCHSIZE):
        #for i in range(0,1,nn.BATCHSIZE):
            #create matricies of training data and labels
            data = train_images[i:i+nn.BATCHSIZE]
            labels = train_labels[i:i+nn.BATCHSIZE]

            # wrap them in Variable
            (data, labels) = pt.autograd.Variable(data), pt.autograd.Variable(labels)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            
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
        print("Loading pytorch data.")
        #Numpy files are much faster to load than the mnist data
        #Try to load these first
        train_images = pt.load("data/pytorch/train_images.pt")
        train_labels = pt.load("data/pytorch/train_labels.pt")
        test_images = pt.load("data/pytorch/test_images.pt")
        test_labels = pt.load("data/pytorch/test_labels.pt")
        wmu_test_images = pt.load("data/pytorch/wmu_test_images.pt")
        wmu_test_labels = pt.load("data/pytorch/wmu_test_labels.pt")

    except IOError as e1:
        print(e1)
        print("Pytorch hasn't been written. Trying idx...")

        try:
            #Load the data from mnist
            np_train_images = cparser.idx("data/mnist/train_images.idx")
            np_train_labels = cparser.idx("data/mnist/train_labels.idx")
            np_test_images = cparser.idx("data/mnist/test_images.idx")
            np_test_labels = cparser.idx("data/mnist/test_labels.idx")
            np_wmu_test_images = cparser.idx("data/wmu/wmu_test_images.idx")
            np_wmu_test_labels = cparser.idx("data/wmu/wmu_test_labels.idx")

            #images come back as a 3d np array.
            #reshpae the image data, normalize it(retypes to floats), and append 1's for bias
            shape = np_train_images.shape
            np_train_images = (np_train_images.reshape(shape[0], 28*28))/255
            
            shape = np_test_images.shape
            np_test_images = (np_test_images.reshape(shape[0], 28*28))/255

            shape = np_wmu_test_images.shape
            np_wmu_test_images = (np_wmu_test_images.reshape(shape[0], 28*28))/255

            train_images = pt.from_numpy(np_train_images)
            train_labels = pt.from_numpy(np_train_labels)
            test_images = pt.from_numpy(np_test_images)
            test_labels = pt.from_numpy(np_test_labels)
            wmu_test_images = pt.from_numpy(np_wmu_test_images)
            wmu_test_labels = pt.from_numpy(np_wmu_test_labels)

            #Make the mnpy directory if it doesn't exist.
            if not os.path.exists("data/pytorch"):
                os.makedirs("data/pytorch")

            pt.save(train_images, "data/pytorch/train_images.pt")
            pt.save(train_labels, "data/pytorch/train_labels.pt")
            pt.save(test_images, "data/pytorch/test_images.pt")
            pt.save(test_labels, "data/pytorch/test_labels.pt")
            pt.save(wmu_test_images, "data/pytorch/wmu_test_images.pt")
            pt.save(wmu_test_labels, "data/pytorch/wmu_test_labels.pt")

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
    
    
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels, wmu_test_images, wmu_test_labels = load_data()
    
    if pt.cuda.is_available():
       train_images = train_images.cuda()
       train_labels = train_labels.cuda()
       test_images = test_images.cuda()
       test_labels = test_labels.cuda()
       wmu_test_images = wmu_test_images.cuda()
       wmu_test_labels = wmu_test_labels.cuda()

    train_start = time.time()
    train(train_images, train_labels, validate_images, validate_labels)
    train_end = time.time()
    print("Total time to train:", train_end-train_start)
    test(test_images, test_labels, wmu_test_images, wmu_test_labels)













