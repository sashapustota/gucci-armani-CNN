import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="Name of the image file in the data folder")
    args = parser.parse_args()
    return args

def predict(image):

    # Load the saved model
    model = tf.keras.models.load_model('model.h5')

    # Load and preprocess the image
    image = Image.open('pred/' + image)

    # Resize the image
    image = image.resize((224, 224))

    image = np.array(image) / 255.0

    # Make predictions
    predictions = model.predict(np.expand_dims(image, axis=0))

    # Get the indices of the top 5 predicted classes
    top5_indices = np.argsort(predictions[0])[::-1][:13]

    top5_confidences = predictions[0][top5_indices]

    # # Get the corresponding class labels
    top5_labels = [labels[i] for i in top5_indices]

    for label, confidence in zip(top5_labels, top5_confidences):
        print(f"{label}: {confidence}")

def main():
    args = parse_args()
    predict(args.image)

if __name__== "__main__":
    main()
