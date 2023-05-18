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
    def forward(self, inputs):
        # Implement the ReLU formula
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, b_input):
        # Implement the ReLU derivative with respect to the input
        derivative = np.where(self.inputs > 0, 1, 0)
        return derivative * b_input


class Sigmoid:
    def forward(self, inputs):
        # Implement the sigmoid formula
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs

    def backward(self, b_input):
        # Implement the sigmoid derivative with respect to the input
        derivative = self.outputs * (1 - self.outputs)
        return derivative * b_input


class Softmax:
    def forward(self, inputs):
        # Implement the softmax formula
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.outputs

    def backward(self, b_input):
        # Implement the softmax derivative with respect to the input
        batch_size = b_input.shape[0]
        jacobian_matrix = np.zeros((batch_size, b_input.shape[1], b_input.shape[1]))
        for i in range(batch_size):
            for j in range(b_input.shape[1]):
                for k in range(b_input.shape[1]):
                    if j == k:
                        jacobian_matrix[i][j][k] = self.outputs[i][j] * (1 - self.outputs[i][k])
                    else:
                        jacobian_matrix[i][j][k] = -self.outputs[i][j] * self.outputs[i][k]
        return np.matmul(b_input[:, np.newaxis, :], jacobian_matrix).squeeze()


class Categorical_Cross_Entropy_loss:
    def forward(self,softmax_output,class_label):
        num_samples = class_label.shape[0]
        loss = -np.log(softmax_output[range(num_samples),class_label]).mean()
        return loss

    def backward(self,softmax_output,class_label):
        num_samples = class_label.shape[0]
        grad = softmax_output
        grad[range(num_samples),class_label] -= 1
        grad /= num_samples
        return grad


class SGD:
    def __init__(self,learning_rate = 0.001):
        self.learning_rate = learning_rate
        
    def update(self,layer):
        layer.weights -= self.learning_rate * layer.grad_wrt_weights
        layer.bias -= self.learning_rate * layer.grad_wrt_bias

        
class Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with random values and biases with zeros
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        # Compute the dot product of input and weight matrix, add bias, and apply activation function
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, b_input):
        # Compute gradients of weights and biases
        inputs = b_input.T
        self.weight_gradients = np.dot(inputs, self.output)
        self.bias_gradients = np.sum(inputs, axis=0)
        return self.weight_gradients, self.bias_gradients


feature_extractor = resnet34(pretrained=True)
num_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Identity()


Layer1 = Dense(num_features, 20)
Act1 = ReLU()
Layer2 = Dense(20, 10)
Act2 = Softmax()

# summary(feature_extractor)


Loss = Categorical_Cross_Entropy_loss()
Optimizer = SGD(learning_rate=0.001)

# Freeze all layers except the last layer
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.fc.requires_grad = True


batch_size = 32
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