import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
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
# x_train = train_data.data
# y_train = train_data.targets

batch_size = 32

trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        # Implement the ReLU formula
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, b_input):
        # Implement the ReLU derivative with respect to the input
        derivative = np.where(self.inputs > 0, 1, 0)
        self.b_output = derivative * b_input
        return self.b_output


class Sigmoid:
    def forward(self, inputs):
        # Implement the sigmoid formula
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, b_input):
        # Implement the sigmoid derivative with respect to the input
        derivative = self.output * (1 - self.output)
        self.b_output = derivative * b_input
        return self.b_output


class Softmax:
    def forward(self, inputs):
        # Implement the softmax formula
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        return self.output

    def backward(self, b_input):
        # Implement the softmax derivative with respect to the input
        batch_size = b_input.shape[0]
        jacobian_matrix = np.zeros((batch_size, b_input.shape[1], b_input.shape[1]))
        for i in range(batch_size):
            for j in range(b_input.shape[1]):
                for k in range(b_input.shape[1]):
                    if j == k:
                        jacobian_matrix[i][j][k] = self.output[i][j] * (1 - self.output[i][k])
                    else:
                        jacobian_matrix[i][j][k] = -self.output[i][j] * self.output[i][k]
        self.b_output = np.matmul(b_input[:, np.newaxis, :], jacobian_matrix).squeeze()
        return self.b_output


class Categorical_Cross_Entropy_loss:
    def forward(self,softmax_output,class_label):
        num_samples = class_label.shape[0]
        epsilon = 1e-10 # To prevent division by zero errors
        loss = -np.log(softmax_output[range(num_samples),class_label] + epsilon).mean()
        self.output = loss
        return self.output

    def backward(self,softmax_output, class_label):
        one_hot_encoded = np.zeros((len(softmax_output), len(softmax_output[0])))
        for i, row in enumerate(one_hot_encoded):
          row[class_label[i]] = 1
        self.b_output = - one_hot_encoded / softmax_output
        return self.b_output


class SGD:
    def __init__(self,learning_rate = 0.001):
        self.learning_rate = learning_rate
        
    def update(self,layer):
        layer.weights -= self.learning_rate * layer.b_weights
        layer.biases -= self.learning_rate * layer.b_biases

        
class Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with random values and biases with zeros
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        # Compute the dot product of input and weight matrix, add bias, and apply activation function
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, b_input):
        # Compute gradients of weights and biases
        self.b_weights = np.dot(self.inputs.T, b_input)
        self.b_biases = np.sum(b_input, axis=0, keepdims=False)
        self.b_output = np.dot(b_input, self.weights.T)
        return self.b_output


feature_extractor = resnet34(pretrained=True)
num_features = feature_extractor.fc.in_features
feature_extractor.fc = nn.Identity()


Layer1 = Dense(num_features, 20)
Act1 = ReLU()
Layer2 = Dense(20, 10)
Act2 = Softmax()

# summary(feature_extractor, (32,32,3))


Loss = Categorical_Cross_Entropy_loss()
Optimizer = SGD(learning_rate=0.001)

# Freeze all layers except the last layer
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.fc.requires_grad = True
maximum = 0
y_predicted_plot = []
y_train_plot = []
#Main Loop of Training
for epoch in range(20):
  # loss = 0
  c = 0
  for i, (x_train, y_train) in enumerate(trainloader):
    # Convert the input image to a PyTorch tensor
    input_tensor = torch.tensor(x_train).float()
    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)
    output = feature_extractor(*input_tensor)

    #forward
    Layer1.forward(output)
    Act1.forward(Layer1.output)
    Layer2.forward(Act1.output)
    Act2.forward(Layer2.output)
    loss = Loss.forward(Act2.output,y_train)
    
    # Report
    y_predict = list(np.argmax(Act2.output,axis = 1))
    counter = 0
    for i in range(len(y_train)):
      y_train_plot.append(int(y_train[i]))
      y_predicted_plot.append(y_predict[i])
      if int(y_train[i]) == y_predict[i]:
        counter += 1
    accuracy = (counter / batch_size) * 100
    if accuracy > maximum:
      maximum = accuracy
    print(f'epoch: {epoch}, batch: {c} of {50000//batch_size}, {accuracy}%, max: {maximum}%')
    print(f'Loss: {loss}')
    print('--------------------------')
    c += 1
    # print(f'Epoch:{epoch}')
    # print(f'Accuracy: {accuracy}')
    
    #backward
    Loss.backward(Act2.output,y_train)
    Act2.backward(Loss.b_output)
    Layer2.backward(Act2.b_output)
    Act1.backward(Layer2.b_output)
    Layer1.backward(Act1.b_output)
    
    #update params
    Optimizer.update(Layer1)
    Optimizer.update(Layer2)

print('--------------------------------------------TESTING PHASE--------------------------------------------')

maximum = 0
c = 0
y_predicted_plot_t = []
y_test_plot_t = []

# Loop of testing
for i, (x_test, y_test) in enumerate(testloader):
    # Convert the input image to a PyTorch tensor
    input_tensor = torch.tensor(x_test).float()
    # Add a batch dimension to the input tensor
    input_tensor = input_tensor.unsqueeze(0)
    output = feature_extractor(*input_tensor)

    #forward
    Layer1.forward(output)
    Act1.forward(Layer1.output)
    Layer2.forward(Act1.output)
    Act2.forward(Layer2.output)
    loss = Loss.forward(Act2.output,y_test)
    
    # Report
    y_predict = list(np.argmax(Act2.output,axis = 1))
    counter = 0
    for i in range(len(y_test)):
      y_test_plot_t.append(int(y_test[i]))
      y_predicted_plot_t.append(y_predict[i])
      if int(y_train[i]) == y_predict[i]:
        counter += 1
    accuracy = (counter / batch_size) * 100
    if accuracy > maximum:
      maximum = accuracy
    print(f'epoch: {epoch}, batch: {c} of {50000//batch_size}, {accuracy}%, max: {maximum}%')
    print(f'Loss: {loss}')
    print('--------------------------')
    
#Confusion Matrix for the training set
cm_train = confusion_matrix(y_train_plot, y_predicted_plot)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_train, annot = True, fmt = 'g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the training set")
plt.show()

#Confusion Matrix for the test set
cm_test = confusion_matrix(y_test_plot_t, y_predicted_plot_t)
plt.subplots(figsize=(10, 6))
sb.heatmap(cm_test, annot = True, fmt = 'g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for the testing set")
plt.show()
