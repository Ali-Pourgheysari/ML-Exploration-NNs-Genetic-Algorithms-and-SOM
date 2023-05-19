import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torchvision import datasets, transforms

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Prepare dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the model architecture
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=1, neurons_per_layer=10, activation='ReLU'):
        super(MyNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        
        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.Linear(input_size, neurons_per_layer),
            nn.ReLU()
        ])
        
        for i in range(hidden_layers - 1):
            self.layers.extend([
                nn.Linear(neurons_per_layer, neurons_per_layer),
                nn.ReLU()
            ])
        
        self.layers.append(nn.Linear(neurons_per_layer, 10))
        
        if activation == 'ReLU':
            self.activation_fn = nn.ReLU()
        else:
            self.activation_fn = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                x = self.activation_fn(x)
        
        return x

# Define training function
def train(model, train_loader, optimizer, criterion):
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        train_loss += loss.item()
        _, predictions = torch.max(outputs.data, 1)
        train_correct += (predictions == targets).sum().item()
        
        loss.backward()
        optimizer.step()
    
    # Calculate training accuracy and loss
    train_accuracy = 100. * train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    
    return train_accuracy, train_loss

# Define testing function
def test(model, test_loader, criterion):
    model.eval()
    
    test_loss = 0
    test_correct = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            test_loss += criterion(outputs, targets).item()
            _, predictions = torch.max(outputs.data, 1)
            test_correct += (predictions == targets).sum().item()
        
    # Calculate testing accuracy and loss
    test_accuracy = 100. * test_correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    
    return test_accuracy, test_loss



# Define genetic algorithm functions
def generate_individual():
    # Hidden layer count
    num_layers = np.random.randint(3)
    
    # Neuron count per layer
    neurons_per_layer = np.random.choice([10, 20, 30])
    
    # Activation function
    activation_fn = np.random.choice(['ReLU', 'Sigmoid'])
    
    return [num_layers, neurons_per_layer, activation_fn]

def generate_population(population_size):
    population = []
    for i in range(population_size):
        population.append(generate_individual())
    return population

def fitness(individual, model, train_loader, test_loader, criterion, epochs=5):
    # Generate a new model based on individual
    num_layers, neurons_per_layer, activation_fn = individual
    model = MyNetwork(input_size=3072, 
                      hidden_layers=num_layers, 
                      neurons_per_layer=neurons_per_layer, 
                      activation=activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        train_accuracy, train_loss = train(model, train_loader, optimizer, criterion)
        test_accuracy, test_loss = test(model, test_loader, criterion)
        
    # Return the average testing accuracy over 5 runs
    accuracy = 0
    for i in range(5):
        test_accuracy, _ = test(model, test_loader, criterion)
        accuracy += test_accuracy
    accuracy /= 5
    
    return accuracy

def select_parents(population, k=2):
    parents = []
    fitnesses = {}
    
    # Calculate fitness for all individuals
    for individual in population:
        fitnesses[str(individual)] = fitness(individual, model, train_loader, test_loader, criterion)
    
    # Select the k individuals with highest fitness
    for i in range(k):
        fittest_individual = max(fitnesses, key=fitnesses.get)
        parents.append(eval(fittest_individual))
        del fitnesses[fittest_individual]
    
    return parents

def crossover(parents):
    offspring = []
    
    # Perform uniform crossover
    for i in range(len(parents[0])):
        if np.random.uniform() < 0.5:
            offspring.append(parents[0][i])
        else:
            offspring.append(parents[1][i])
    
    return offspring

def mutation(individual, mutation_rate=0.1):
    # Mutate a random gene with probability mutation_rate
    num_layers, neurons_per_layer, activation_fn = individual
    if np.random.uniform() < mutation_rate:
        num_layers = np.random.randint(3)
    if np.random.uniform() < mutation_rate:
        neurons_per_layer = np.random.choice([10, 20, 30])
    if np.random.uniform() < mutation_rate:
        activation_fn = np.random.choice(['ReLU', 'Sigmoid'])
    
    return [num_layers, neurons_per_layer, activation_fn]

def evolve(population, model, train_loader, test_loader, criterion, popSize=10, k=2, mutation_rate=0.1):
    new_population = []
    
    # Select two parents and generate two offspring until population is filled
    while len(new_population) < popSize:
        parents = select_parents(population, k=k)
        offspring1 = mutation(crossover(parents), mutation_rate=mutation_rate)
        offspring2 = mutation(crossover(parents), mutation_rate=mutation_rate)
        new_population.append(offspring1)
        new_population.append(offspring2)
    
    return new_population

# Set hyperparameters
epochs = 5
popSize = 10
k = 2
mutation_rate = 0.1

# Generate initial population
population = generate_population(population_size=popSize)

# Train the model using genetic algorithm
model = MyNetwork(input_size=3072, hidden_layers=1, neurons_per_layer=10, activation='ReLU')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for generation in range(10):
    print(f"Generation {generation}")
    population_fitnesses = []
    
    for individual in population:
        fitness_score = fitness(individual, model, train_loader, test_loader, criterion)
        population_fitnesses.append((individual, fitness_score))
    
    population_fitnesses = sorted(population_fitnesses, key=lambda x: x[1], reverse=True)
    best_individual, best_fitness = population_fitnesses[0]
    
    print(f"Best individual: {best_individual}, Fitness score: {best_fitness}")
    
    population = evolve(population, model, train_loader, test_loader, criterion, popSize=popSize, k=k, mutation_rate=mutation_rate)
    
    print("=" * 40)

print(f"Best individual: {best_individual}, Fitness score: {best_fitness}")