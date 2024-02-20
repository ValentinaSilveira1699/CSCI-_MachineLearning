import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a KNeighborsClassifier
clf = KNeighborsClassifier()

# Split the data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test set
predicted = clf.predict(X_test)

# Initialize a figure
plt.figure(figsize=(10,5))

# Display the first incorrectly predicted sample image in each of the 10 classes
for i in range(10):
    # Find the indices of incorrectly predicted samples for class i
    wrong_indices = [index for index, (true_label, pred_label) in enumerate(zip(y_test, predicted))
                     if true_label == i and pred_label != i]

    # If there are incorrectly predicted samples for class i
    if wrong_indices:
        # Take the first incorrect prediction for class i
        wrong_index = wrong_indices[0]
        wrong_image = X_test[wrong_index].reshape(8, 8)
        true_label = y_test[wrong_index]
        pred_label = predicted[wrong_index]

        # Plot the image
        plt.subplot(2, 5, i + 1)
        plt.imshow(wrong_image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"True: {true_label}\nPred: {pred_label}")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('incorrectly_predicted_samples.png')
plt.show()
