import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

# Load the digits dataset
digits = datasets.load_digits()

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, random_state=42)

# Train a SVM classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)

# Make predictions on the test set
predicted = classifier.predict(X_test)

# Identify incorrectly predicted samples
incorrect_predictions = (predicted != y_test)

# Display the first 4 incorrectly predicted test samples
count = 0
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, correct_label, wrong_label in zip(axes, X_test[incorrect_predictions], y_test[incorrect_predictions], predicted[incorrect_predictions]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Correct: {correct_label}\nPrediction: {wrong_label}")
    count += 1
    if count == 4:
        break

# Save the figure
plt.savefig("fourDigits.png")
plt.show()
