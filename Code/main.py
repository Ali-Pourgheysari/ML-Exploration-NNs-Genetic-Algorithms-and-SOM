import numpy as np
import matplotlib.pyplot as plt
# torch is just for the feature extractor and the dataset (NOT FOR IMPLEMENTING NEURAL NETWORKS!)
import torch
from torchsummary import summary
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import resnet34
# sklearn is just for evaluation (NOT FOR IMPLEMENTING NEURAL NETWORKS!)
from sklearn.metrics import confusion_matrix, f1_score

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
# You should define x_train and y_train
x_train = train_data.data
y_train = train_data.targets

# batch_size = 32

# trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

class ReLU:
    def forward(self,inputs):
        # // To do: Implement the ReLU formula

    def backward(self,b_input):
        # // To do: Implement the ReLU derivative with respect to the input

class Sigmoid:
    def forward(self,inputs):
        # // To do: Implement the sigmoid formula

    def backward(self,b_input):
        # // To do: Implement the sigmoid derivative with respect to the input

class Softmax:
    def forward(self,inputs):
        # // To do: Implement the softmax formula
    
    def backward(self,b_input):
        # // To do: Implement the softmax derivative with respect to the input

class Categorical_Cross_Entropy_loss:
    def forward(self,softmax_output,class_label):
        # // To do: Implement the CCE loss formula

    def backward(self,softmax_output,class_label):
        # // To do: Implement the CCE loss derivative with respect to predicted label

class SGD:
    def __init__(self,learning_rate = 0.001):
        self.learning_rate = learning_rate
        
    def update(self,layer):
        # // To do: Update layer params based on gradient descent rule
        
class Dense:
    def __init__(self,n_inputs,n_neurons):
        # // To do: Define initial weight and bias
    
    def forward(self,inputs):
        # // To do: Define input and output

    def backward(self,b_input):
        # // To do: Weight and bias gradients

feature_extractor = resnet34(pretrained=True)
Layer1 = Dense(d,20) # d is the output dimension of feature extractor
Act1 = ReLU()
Layer2 = Dense(20,10)
Act2 = Softmax()
Loss = Categorical_Cross_Entropy_loss()
Optimizer = SGD(learning_rate=0.001)

#Main Loop of Training
for epoch in range(20):
    #forward
    Layer1.forward(x_train)
    Act1.forward(Layer1.output)
    Layer2.forward(Act1.output)
    Act2.forward(Layer2.output)
    loss = Loss.forward(Act2.output,y_1hot)
    
    # Report
    y_predict = np.argmax(Act2.output,axis = 1)
    accuracy = np.mean(y_train == y_predict)
    print(f'Epoch:{epoch}')
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print('--------------------------')
    
    #backward
    Loss.backward(Act2.output,y_1hot)
    Act2.backward(Loss.b_output)
    Layer2.backward(Act2.b_output)
    Act1.backward(Layer2.b_output)
    Layer1.backward(Act1.b_output)
    
    #update params
    Optimizer.update(Layer1)
    Optimizer.update(Layer2)

#Confusion Matrix for the training set
cm_train = confusion_matrix(y_train, y_predict)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_train, annot = True, fmt = 'g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the training set")
plt.show()

#Confusion Matrix for the test set
# // To Do