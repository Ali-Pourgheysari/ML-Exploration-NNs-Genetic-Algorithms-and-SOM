# Neural Network Implementation, Genetic Algorithm, and Self-Organizing Maps (SOM)
This project consists of three parts that explore different aspects of machine learning and artificial neural networks.

## Part 1: Neural Network Implementation
In this part, a simple Neural Network is implemented from scratch using PyTorch for deep learning framework to extract features. The network contains custom implementations of activation functions (ReLU and Sigmoid), a Softmax layer, Categorical Cross-Entropy loss function, and a Stochastic Gradient Descent optimizer. The network is trained on the CIFAR-10 dataset, and the training process and performance are displayed. The network's architecture and hyperparameters can be customized, allowing you to experiment with different configurations.

## Part 2: Genetic Algorithm for Neural Network Architecture Search
In this part, a Genetic Algorithm is utilized to perform a neural network architecture search. The algorithm generates and evaluates multiple neural network architectures with varying numbers of hidden layers, neurons per layer, and activation functions. The performance of each architecture is measured using the CIFAR-10 dataset. The algorithm then evolves the population of architectures over multiple generations, selecting the best-performing ones. This process allows the algorithm to discover better neural network architectures through genetic evolution.

## Part 3: Self-Organizing Maps (SOM) for Feature Visualization
In this part, a pre-trained ResNet34 model is used to extract feature vectors from the CIFAR-10 dataset. These feature vectors are then visualized using a Self-Organizing Map (SOM) network, implemented using the MiniBatchKMeans algorithm. The SOM network clusters the feature vectors into different groups and visualizes the weight vectors for the final feature mapping. Additionally, the algorithm reports the scatter of different labels in each cluster, providing insights into how the feature vectors are grouped according to their labels.

## Requirements
* Python 3.x
* PyTorch (for Part 1)
* NumPy
* Matplotlib
* Seaborn
* TorchVision (for Part 2)
* Scikit-learn (for Part 3)

## How to Use
1. Install Dependencies: Ensure you have Python 3.x installed. Install the required libraries using the following command:
```bash
pip install torch numpy matplotlib seaborn torchvision scikit-learn
```
2. Run the Scripts: Execute each script separately based on the parts you want to explore:
* `part1.py`: Run this script to implement a simple neural network from scratch and train it on the CIFAR-10 dataset. You can modify the network architecture and hyperparameters to experiment with different configurations.

* `part2.py`: Execute this script to perform a neural network architecture search using a Genetic Algorithm. The script will generate and evaluate various network architectures and evolve them over multiple generations to discover better configurations.

* `part3.py`: Run this script to extract feature vectors from the CIFAR-10 dataset using a pre-trained ResNet34 model and visualize the feature vectors using a Self-Organizing Map (SOM) network. The script will cluster the feature vectors and report the scatter of different labels in each cluster.

Note: The datasets required for Part 1 and Part 2 (CIFAR-10) will be automatically downloaded during script execution.

## Credits
This project uses the CIFAR-10 dataset and incorporates concepts from Neural Networks, Genetic Algorithms, and Self-Organizing Maps. The implementations provided are for educational purposes and serve as examples of how these machine learning techniques can be applied in different scenarios.

 Read the complete documentation [HERE](documentation.pdf).
Also you can read [THIS](Code_review.pdf) code review for more details about the code.