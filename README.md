🧠 QuickDraw CNN Classifier
A Convolutional Neural Network (CNN) trained to recognize doodles from the Quick, Draw! dataset. The project includes training code, model export, and an interactive drawing app to test the model locally.

📁 Project Structure
keys.py – Contains the list of categories to include in the model.

loadData.py – Downloads and prepares data from the web.

modelTraining.py – Trains the CNN and saves the model as QuickDraw_CNN_35_classes.h5.

QuickDraw_CNN_35_classes.h5 – The trained CNN model (output file).

doodle_app.py – A simple Pygame-based app to draw and classify doodles using the trained model. This file must be in the same directory as the saved model.

🚀 How to Train the Model Yourself
Open Google Colab.

Upload the following files:

keys.py

loadData.py

modelTraining.py
(Drag and drop the files into the left-side file panel.)

Run the training script using:

python
Copy
Edit
!python modelTraining.py
Download the output file: QuickDraw_CNN_35_classes.h5.

Place QuickDraw_CNN_35_classes.h5 in the same folder as doodle_app.py.

🎮 Running the App Locally
Requirements:

Python 3.11.0 or lower (due to TensorFlow compatibility)

Install the following libraries:

bash
Copy
Edit
pip install pygame
pip install tensorflow
pip install numpy
pip install Pillow
Then run:

bash
Copy
Edit
python doodle_app.py
Use the canvas to draw a doodle — the model will predict what you drew!

⚙️ Tips for Customization
Feel free to experiment with:

Number of categories (keys.py)

Model architecture and hyperparameters (modelTraining.py)

Training duration and dataset size

📊 Current Model Performance
Test Accuracy: 92.56%
(Using 35 categories from the QuickDraw dataset)

📝 Notes
The model uses a simple CNN architecture suitable for quick training and good performance on 28x28 grayscale images.

Data is fetched from the official QuickDraw dataset hosted on Google's servers.
