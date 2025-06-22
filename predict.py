# predict.py
# This script loads the trained model and makes a prediction on a random test image.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keys
import random
import os

# --- Configuration ---
MODEL_PATH = "QuickDraw_CNN_35_classes.h5"
TEST_IMAGES_PATH = "test_images.npy"
TEST_LABELS_PATH = "test_labels.npy"

def run_prediction():
    """Loads model and data, then predicts on a random image."""
    # 1. Check if model and data exist
    if not all(os.path.exists(p) for p in [MODEL_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH]):
        print("Error: Model or test data not found.")
        print("Please run 'modelTraining.py' first to train and save the necessary files.")
        return

    # 2. Load the trained model, test data, and class names
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = keys.get_keys()
    
    x_test = np.load(TEST_IMAGES_PATH)
    y_test = np.load(TEST_LABELS_PATH) # Integer labels

    # 3. Select a random image from the test set
    random_index = random.randint(0, len(x_test) - 1)
    image_to_predict = x_test[random_index]
    actual_label_index = y_test[random_index]
    actual_label_name = class_names[actual_label_index]
    
    # 4. Make a prediction
    # The model expects a batch of images, so we add an extra dimension.
    image_for_model = np.expand_dims(image_to_predict, axis=0)
    prediction = model.predict(image_for_model)
    
    # The prediction is an array of probabilities. Find the highest one.
    predicted_index = np.argmax(prediction)
    predicted_name = class_names[predicted_index]
    confidence = np.max(prediction) * 100

    # 5. Display the results
    print(f"\nPrediction for a random test image:")
    print(f"  -> Actual Label:    '{actual_label_name}'")
    print(f"  -> Predicted Label: '{predicted_name}'")
    print(f"  -> Confidence:      {confidence:.2f}%")
    
    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_predict.squeeze(), cmap='gray_r') # Use gray_r for black ink
    plt.title(f"Actual: {actual_label_name}\nPredicted: {predicted_name}", fontsize=12)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_prediction()

