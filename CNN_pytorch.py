import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from skimage import io
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Parameter1-> input image channel, Parameter2-> output channels, 3X3 square convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # an affine operation: y = Wx + b
        #number of nodes on flattening 
        self.fc1 = nn.Linear(2048, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #print(self.num_flat_features(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = F.softmax(self.fc4(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#Train function for training the model of CNN
def train(net, train_img, train_lbls, validation_img, validation_lbls, test_img):
		max_epochs=30
		n_batches=156
		train_size=1252
		batch_size=8

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		# making mini- batches of the the input data
		for i in range(max_epochs):
			residual=0.0
			for j in range(n_batches):
				(batch_x, batch_y)= train_img[j*batch_size:(j+1)*batch_size], train_lbls[j*batch_size:(j+1)*batch_size]
				#convert the vector in pytorch tensor Variables
				batch_x=np.array(batch_x, dtype=np.float64)
				batch_y=np.array(batch_y, dtype=np.float64)
				#print(type(batch_x), batch_x.dtype, batch_x.shape)
				batch_x=Variable(torch.from_numpy(batch_x).double(), requires_grad=True)
				batch_y=Variable(torch.from_numpy(batch_y).long())

				optimizer.zero_grad()
				outputs = net(batch_x)
				loss = criterion(outputs, batch_y)
				loss.backward()
				optimizer.step()
				#print(loss)

				#residual += loss.item()
				print('epoch: %d' %(i+1), 'batch: %d' %(j+1), 'loss: %f' %loss)
				residual=0
		print("FINISHED TRAINING THE MODEL ..........................")

		#Validating the model .................................................
		print("VALIDATING THE MODEL .................................")
		validation_img = Variable(torch.from_numpy(np.array(validation_img, dtype=np.float64)).double(), requires_grad=False)
		validation_lbls = Variable(torch.from_numpy(np.array(validation_lbls)).long(), requires_grad=False)
		loss = criterion(net(validation_img), validation_lbls)
		print("Validation loss = %f" %loss)
		output = net(validation_img) #F.softmax(net(validation_img), dim=1)
		correct_count= 0;
		wrong_count= 0
		for out in range(len(output)):
			if ((output[out][0]>=0.5 and validation_lbls[out] ==0) or (output[out][0]<0.5 and validation_lbls[out] ==1)):
				correct_count+=1
			else:
				wrong_count+=1
		print("The accuracy is %d %%" %correct_count)

		#saving the model .......................................................
		print('MODEL BEING SAVED ..............................')
		torch.save(net, '/home/swapnil/Desktop/CNN_vision/model.pth')

		#preedicting the model ....................................................
		print("PREDICTING THE MODEL ...............................")
		test_img = Variable(torch.from_numpy(test_img).double(), requires_grad=False)
		outputs=net(test_img)
		outputs = F.softmax(outputs, dim=1)
		#SOFTMAX LAYER HAS TO USED
		prediction=list()
		for out in range(len(outputs)):
			if outputs[out][0]>=0.5:
				prediction.append(1)
			else:
				prediction.append(0)
		return prediction

#Loading training and test data
train_size=1252
dim_input=32*32
dim_output=2

train_img= list()
train_lbls= list()


print("Loading Training Data ........................")

for i in range(train_size):
	name='/home/swapnil/Desktop/CNN_vision/train/img_' + str(i+1) +'.jpg'
	img=io.imread(name)
	img=np.reshape(img, (1,32,32))
	img=img.astype(float)
	train_img.append(img)
	#Odd name is collar and even name in round
	#class '1' is for Round and Class '0' is for V_neck Tshirt or Collar
	if (i+1) % 2==0:
		train_lbls.append(1)
	else:
		train_lbls.append(0)
	#if(i%500==0):
		#print('Image %d is loaded' %i)

#Loading Validation data

validation_img= list()
validation_lbls= list()

validation_size= 100
dim_input=32*32
dim_output=2

print("Loading Validation Data ........................")

for i in range(validation_size):
	name='/home/swapnil/Desktop/CNN_vision/validation/img_' + str(i+1) +'.jpg'
	img=io.imread(name)
	img=np.reshape(img, (1,32,32))
	img=img.astype(float)
	validation_img.append(img)
	#Odd name is collar and even name in round
	#class '1' is for Round and Class '0' is for V_neck Tshirt or Collar
	if (i+1) % 2==0:
		validation_lbls.append(1)
	else:
		validation_lbls.append(0)


#Loading the test exmaples
test_img=list()
test_size=24
test2=list() #Buffer images for use in last line of the code
print("Loading Test Data .............................")
for i in range(test_size):
	name= '/home/swapnil/Desktop/CNN_vision/test/img_' + str(i+1) + '.jpg'
	img=io.imread(name)
	img=np.reshape(img, (1,32,32))
	img=img.astype(np.float64)
	test_img.append(img)
	test2.append(io.imread(name))
test_img= np.array(test_img, dtype=np.float64)


print("Starting the training ........................")
# object declaration here and actual function call
net=Net()
net.double() #to be looked for ************************
prediction= train(net, train_img, train_lbls, validation_img, validation_lbls, test_img) # 1 in prediction for Collar and 0 for V_neck
print("Image saved in respective folder in prediction .........................")
#Save image based on classes in respective folder
for i in range(len(prediction)):
	if prediction[i]==1:
		name='/home/swapnil/Desktop/CNN_vision/prediction/Round/' + str(i+1) + '.jpg'
	else:
		#name='/home/swapnil/Desktop/CNN_vision/train/prediction/V_neck/' + str(i+1) + '.jpg'
		name='/home/swapnil/Desktop/CNN_vision/prediction/Collar/' + str(i+1) + '.jpg'
	plt.imsave(name, test2[i])   #save image at the location
