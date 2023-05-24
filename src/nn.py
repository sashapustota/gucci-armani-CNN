import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-trs', '--train_samples', type=int, default=50000)
    parser.add_argument('-vs', '--val_samples', type=int, default=7000)
    parser.add_argument('-tes', '--test_samples', type=int, default=2000)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-es', '--early_stop', type=int, default=5)
    return parser.parse_args()

def load_data(batch_size, train_samples, val_samples, test_samples):

    # Load the JSON file into a dictionary
    with open('image_data.json', 'r') as f:
        data_dict = json.load(f)
    
    # Transform the dictionary into a DataFrame
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.reset_index(inplace=True)
    df.columns = ['name', 'label', 'path']

    df = df[df['label'] != '1st-optional-photo']
    df = df[df['label'] != '2nd-optional-photo']
    df = df[df['label'] != 'Missing']
    df = df[df['label'] != 'qr-code-label']
    df = df[df['label'] != 'logo-texture-macro-image']
    df = df[df['label'] != 'size-tag']
    df = df[df['label'] != 'insole-front-side']
    df = df[df['label'] != 'insole-back-side']
    df = df[df['label'] != 'inside-stitching']

    # define data generator for training set
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

    # define data generator for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=420)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=420)

    num_train_samples = train_samples
    num_val_samples = val_samples
    num_test_samples = test_samples

    # select a random subset of the data
    train_df = train_df.sample(n=num_train_samples, random_state=420)
    val_df = val_df.sample(n=num_val_samples, random_state=420)
    test_df = test_df.sample(n=num_test_samples, random_state=420)

    # set batch size and image size
    batch_size = 32
    image_size = (224, 224)

    # define the train, validation, and test data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='path',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator, num_train_samples, num_val_samples

def load_model(early_stop_num):
    pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freezing pre-trained layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # New layers for classification
    num_classes = 13
    x = pretrained_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

    # Compile the model with an appropriate loss function, optimizer, and evaluation metric.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_num)

    return model, early_stop

def fit_model(epochs, num_train_samples, num_val_samples, batch_size, train_generator, val_generator, early_stop, model):

    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size
    epochs = epochs

    history = model.fit_generator(train_generator, 
                steps_per_epoch=steps_per_epoch, 
                epochs=epochs, 
                validation_data=val_generator, 
                validation_steps=validation_steps,
                callbacks=[early_stop])
    
    # Save the model
    model.save('model.h5')

    print("Training complete, model saved.")

    return history

def plot_metrics(history):

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy.png')

    # Clear the plot
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss.png')

    print("Plots created!")

def accuracy_report(model, test_generator):
    # Use the model to predict class probabilities for the test set
    preds = model.predict_generator(test_generator, steps=len(test_generator))

    # Save the classification report
    pred = np.argmax(preds, axis=1)
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(test_generator.classes, pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv', index=True)
    print("Created classification report!")

def main():
    args = parse_args()
    train_generator, val_generator, test_generator, num_train_samples, num_val_samples = load_data(args.batch_size, args.train_samples, args.val_samples, args.test_samples)
    model, early_stop = load_model(args.early_stop)
    history = fit_model(args.epochs, num_train_samples, num_val_samples, args.batch_size, train_generator, val_generator, early_stop, model)
    plot_metrics(history)
    accuracy_report(model, test_generator)

if __name__ == "__main__":
    main()
