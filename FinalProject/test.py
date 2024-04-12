import mnist_loader
import network
import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)

    net = network.Network([784, 30, 10])

    for epoch in range(20):
        net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

        # Evaluate accuracy after each epoch
        correct_classified = net.evaluate(test_data)
        total_correct = sum(correct_classified)
        print(f"\nEpoch {epoch}: {total_correct} / 10000")

        # Print accuracy for each class
        for i, correct in enumerate(correct_classified):
            print(f"class {i}: {correct}/{sum(1 for _, label in test_data if label == i)}")

    # Display the first incorrectly classified image for each class
    incorrect_images = [[] for _ in range(10)]
    incorrect_labels = [[] for _ in range(10)]
    for x, y in test_data:
        prediction = np.argmax(net.feedforward(x))
        if prediction != y:
            incorrect_images[y].append(x)
            incorrect_labels[y].append((y, prediction))  # Store correct and incorrect labels



    # Display the images with correct and incorrect labels
    images = []
    for i, img_list in enumerate(incorrect_images):
        if img_list:
            images.append((img_list[0].reshape(28, 28), incorrect_labels[i]))
    combined_image = combine_images(images, incorrect_labels)  # Fix here
    combined_image.show()

    # Save the combined image as "allTen.png"
    combined_image.save("allTen.png")

def combine_images(images, labels):
    width = 56 * len(images)  # Increase image size to 56x56 pixels
    height = 56 * len(images[0])  # Adjust height to accommodate labels
    combined = Image.new('L', (width, height), color=0)  # Set background to white
    font = ImageFont.truetype("arial.ttf", 11)  # Font Size is 11
    for i, img_list in enumerate(images):
        for j, img in enumerate(img_list):
            im = Image.fromarray(np.uint8(img * 255))
            combined.paste(im, (i * 56, j * 56))
            draw = ImageDraw.Draw(combined)
            correct_label, incorrect_label = labels[i][j]

	    # Print correct and incorrect labe below image
            draw.text((i * 56, j * 56 + 42), f"Correct: {correct_label}", fill=255, font=font)
            draw.text((i * 56, j * 56 + 56), f"Incorrect: {incorrect_label}", fill=255, font=font)  
        
            break  # Only print one correct and one incorrect label
    return combined

if __name__ == "__main__":
    main()

