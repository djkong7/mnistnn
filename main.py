from __future__ import print_function
import os, sys, time, torch as pt, numpy as np, idx_parser as cparser

LEARNING_RATE = .001
INPUT_SIZE = 16*5*5
NUM_CLASSES = 10
BATCHSIZE = 100
HIDDEN_LAYER_SIZE = 200
OUTPUT_SIZE = 10

class Net(pt.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #ouput size = (input_dims-kernel_size)/stride + 1        (30-3)/1+1 = 28
        self.conv1 = pt.nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2)
        self.pool = pt.nn.MaxPool2d(kernel_size = 2)
        self.conv2 = pt.nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, padding = 0)
        self.fc1 = pt.nn.Linear(INPUT_SIZE,120)
        self.fc2 = pt.nn.Linear(120,84)
        self.fc3 = pt.nn.Linear(84,10)
        
    def forward(self, x):
        x = self.pool(pt.nn.functional.relu(self.conv1(x)))
        x = self.pool(pt.nn.functional.relu(self.conv2(x)))
        #print(x)
        x = x.view(-1, INPUT_SIZE)
        x = pt.nn.functional.relu(self.fc1(x))
        x = pt.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

def train(net, train_images, train_labels, validate_images, validate_labels):
    num_epoch = 1000
    stop = False
 
    loss = np.Inf
    counter = 0
    loss_increase = 0

    #For calculating loss
    criterion = pt.nn.modules.loss.CrossEntropyLoss()
    optimizer = pt.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = 0.9)
    
    for j in range(0,num_epoch):
        print("Epoch number", j)
        start = time.time()#Just for timing
        for i in range(0,train_images.shape[0], BATCHSIZE):
        #for i in range(0,1,nn.BATCHSIZE):
            #create matricies of training data and labels
            data = train_images[i:i+BATCHSIZE]
            labels = train_labels[i:i+BATCHSIZE]

            # wrap them in Variable
            (data, labels) = pt.autograd.Variable(data), pt.autograd.Variable(labels)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            y = net(data)
            new_loss = criterion(y,labels)
            new_loss.backward()
            optimizer.step()

            counter += 1

            if (counter % 150) == 0:

                #Automatically stop to prevent overtraining.
                if (new_loss.data[0] - loss) <= 0:
                    loss = new_loss.data[0]
                    loss_increase = 0
                    pt.save(net.state_dict(),"model_save/model")
                    #print("Saved!")
                    #print("\tLoss: %f"%(new_loss.data[0]))
                else:
                    loss_increase += 1
                    #print(loss_increase)
                    if loss_increase >= 8:
                        #Stop epoch loop
                        stop = True
                        #Stop batchsize loop
                        break

        end = time.time()
        #print("\tTime for Epoch:",end-start)
        if stop:
            break

def test(net, test_images, test_labels, wmu_test_images, wmu_test_labels):
    test_images = pt.autograd.Variable(test_images)
    y = net(test_images)
    _, predicted = pt.max(y.data,1)
    correct = (predicted == test_labels).sum()
    
    print("MNIST")
    print("\tAccuracy: %f"%(correct/test_labels.shape[0]*100))
    
    wmu_test_images = pt.autograd.Variable(wmu_test_images)
    y = net(wmu_test_images)
    _, predicted = pt.max(y.data,1)
    correct = (predicted == wmu_test_labels).sum()
    
    print("WMU")
    print("\tAccuracy: %f"%(correct/wmu_test_labels.shape[0]*100))

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
            #normalize it
            np_train_images = np_train_images / 255
            np_test_images = np_test_images / 255
            np_wmu_test_images = np_wmu_test_images / 255

            train_images = pt.from_numpy(np_train_images).float().unsqueeze(1)
            train_labels = pt.from_numpy(np_train_labels).long()
            test_images = pt.from_numpy(np_test_images).float().unsqueeze(1)
            test_labels = pt.from_numpy(np_test_labels).long()
            wmu_test_images = pt.from_numpy(np_wmu_test_images).float().unsqueeze(1)
            wmu_test_labels = pt.from_numpy(np_wmu_test_labels).long()

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
    
    net = Net()
    
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels, wmu_test_images, wmu_test_labels = load_data()
    
    
    if pt.cuda.is_available():
        print("On Cuda")
        net.cuda()
        train_images = train_images.cuda()
        train_labels = train_labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()
        wmu_test_images = wmu_test_images.cuda()
        wmu_test_labels = wmu_test_labels.cuda()

    train_start = time.time()
    train(net, train_images, train_labels, validate_images, validate_labels)
    train_end = time.time()
    print("Total time to train:", train_end-train_start)
    
    net.load_state_dict(pt.load("model_save/model"))
    if pt.cuda.is_available():
        net.cuda()
    
    test(net, test_images, test_labels, wmu_test_images, wmu_test_labels)













