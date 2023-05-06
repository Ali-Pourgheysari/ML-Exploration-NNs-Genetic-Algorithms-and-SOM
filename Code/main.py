import 
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