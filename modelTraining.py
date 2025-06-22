# modelTraining.py
# This is the main script to train the CNN model.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Import our custom modules
import keys
import loadData

# --- Configuration ---
SAMPLE_SIZE_PER_CLASS = 20000
EPOCHS = 20
BATCH_SIZE = 256
MODEL_OUTPUT_PATH = "QuickDraw_CNN_35_classes.h5"

def build_model(num_classes):
    """Builds, compiles, and returns the CNN model."""
    model = models.Sequential()
    
    # Input Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten and Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Output Layer - must match the number of classes
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history):
    """Plots the training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)
    
    plt.savefig("training_history.png")
    plt.show()

def main():
    """Main function to run the training process."""
    # 1. Load data
    categories = keys.get_keys()
    num_classes = len(categories)
    x_train, y_train, x_test, y_test = loadData.load_data(categories, SAMPLE_SIZE_PER_CLASS)

    # 2. One-hot encode the labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # 3. Build the model
    model = build_model(num_classes)
    model.summary()
    
    # 4. Train the model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        x_train, y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test_cat),
        verbose=1
    )
    print("--- Model Training Complete ---")
    
    # 5. Evaluate the model
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {acc:.4f} ({acc:.2%})")

    # 6. Save the model and test data
    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    model.save(MODEL_OUTPUT_PATH)
    
    # Save the test data for the prediction script
    print("Saving test data for predict.py...")
    np.save('test_images.npy', x_test)
    np.save('test_labels.npy', y_test) # Save integer labels

    # 7. Plot history
    plot_history(history)

if __name__ == "__main__":
    main()

