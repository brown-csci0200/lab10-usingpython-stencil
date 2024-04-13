# Importing the necessary libraries and modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import idx2numpy
import matplotlib.pyplot as plt

# Load the MNIST dataset

# Load training images and labels
mnist_train_img = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
mnist_train_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')

# Load test images and labels
mnist_test_img = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
mnist_test_labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')

##############################################################

# Visualizing the dataset

# # Print the first 5 images and their labels from the training dataset
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(mnist_train_img[i], cmap='gray')
#     plt.title(f"Label: {mnist_train_labels[i]}")
#     plt.axis('off')

# plt.show()

##############################################################

# Train a RandomForestClassifier on the MNIST dataset

## TODO TASK 5 : UNCOMMENT THE CODE BELOW AND RUN THE PYTHON FILE TO TRAIN THE MODEL

# Create a RandomForestClassifier
# Task 
model = RandomForestClassifier(n_estimators=500, random_state=42) 

# Train the model
model.fit(mnist_train_img.reshape(-1, 28*28), mnist_train_labels)

# Make predictions on the test data
predictions = model.predict(mnist_test_img.reshape(-1, 28*28))

# Evaluate the accuracy of the model
accuracy = accuracy_score(mnist_test_labels, predictions)
print("Accuracy:", accuracy)
