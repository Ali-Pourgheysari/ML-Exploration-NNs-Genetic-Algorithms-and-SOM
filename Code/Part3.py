import torch
import torchvision.models as models
import numpy as np
import torchvision
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# Load the pre-trained ResNet34 model without its final layer
model = models.resnet34(pretrained=True)
model.fc = torch.nn.Identity()

# Load the CIFAR10 dataset without labels
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
cifar_data_loader = torch.utils.data.DataLoader(cifar_dataset, batch_size=64, shuffle=False)

# Extract feature vectors for each image in the dataset using the pre-trained ResNet34
feature_vectors = []
for images, _ in cifar_data_loader:
    with torch.no_grad():
        features = model(images)
    feature_vectors.append(features.numpy())
feature_vectors = np.concatenate(feature_vectors)

som_net = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=100, max_iter=20)

# Train the SOM network
som_net.fit(feature_vectors)

# Plot the weight vectors for the final feature mapping
plt.imshow(som_net.cluster_centers_.reshape(10, -1))
plt.show()

# Report the scatter of the different labels of each cluster
cifar_labels = cifar_dataset.targets
cluster_labels = som_net.predict(feature_vectors)

for i in range(10):
    mask = cluster_labels == i
    mul = [a*b for a,b in zip(mask, cifar_labels)]
    label_counts = np.bincount(mul)
    print(f"Cluster {i}: Label Scatter = {label_counts}")

# B
# som_net = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=100, max_iter=20, init='k-means++', reassignment_ratio=0.01,
#                           max_no_improvement=None, verbose=0, compute_labels=True, n_init=3,
#                           init_size=None, batch_size=100, verbose=0, max_no_improvement=None, reassignment_ratio=0.01,
#                           random_state=None, tol=0.0, max_iter=20, som_topology='P', som_shape=(2, 5), neighborhood_diameter=1)

# C
# som_net = MiniBatchKMeans(n_clusters=10, random_state=0, batch_size=100, max_iter=20, init='k-means++', reassignment_ratio=0.01,
                        #   max_no_improvement=None, verbose=0, compute_labels=True, n_init=3,
                        #   init_size=None, batch_size=100, verbose=0, max_no_improvement=None, reassignment_ratio=0.01,
                        #   random_state=None, tol=0.0, max_iter=20, som_topology='P', som_shape=(2, 5), neighborhood_diameter=1)