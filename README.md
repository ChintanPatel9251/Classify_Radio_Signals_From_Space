# Deep Space Radio Signals Classification
Welcome to the repository for my data analysis project on classifying deep space radio signals. This project focuses on using Keras with TensorFlow to build and train a convolutional neural network (CNN) to classify 2D spectrograms of radio signals collected by the Allen Telescope Array at the SETI Institute.


## Project Description
In this project, you will learn the basics of using Keras with TensorFlow as its backend to solve an image classification problem. The data consists of 2D spectrograms of deep space radio signals, which we will treat as images to train a classification model. By the end of the project, you will have built and trained a CNN from scratch using Keras to classify signals from space into one of four classes.


## Project Structure
The hands-on project on Classifying Radio Signals from Space is divided into the following tasks:

### Task 1: Introduction and Import Libraries
* Introduction to the data and an overview of the project.
* Import essential modules and helper functions from NumPy, Matplotlib, and Keras.

### Task 2: Load and Preprocess SETI Data
* Display 2D spectrograms using Matplotlib.
* Reshape the input data with NumPy.

### Task 3: Create Training and Validation Data Generators
* Generate batches of tensor image data with real-time data augmentation.
* Specify paths to training and validation image directories and generate batches of augmented data.

### Task 4: Create a Convolutional Neural Network (CNN) Model
* Design a CNN with 2 convolution layers and 1 fully connected layer to predict four signal types.
* Use Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.

### Task 5: Learning Rate Scheduling and Compile the Model
* Apply an exponential decay function to the provided initial learning rate.

### Task 6: Train the Model
* Train the CNN by invoking the model.fit() method.
* Use ModelCheckpoint() to save the weights associated with the highest validation accuracy after every epoch.
* Display live training loss and accuracy plots in Jupyter Notebook using livelossplot.

### Task 7: Evaluate the Model
* Evaluate the CNN by invoking the model.evaluate() method.
* Obtain the classification report to note the precision and recall of your classifier.


## Repository Structure
* **notebook/:** Contains the Jupyter notebook with the main project code and analysis.
* **dataset/:** Contains the dataset of 2D spectrograms used for training and testing the model.


## Getting Started
### Prerequisites
* Python 3.x
* Jupyter Notebook
* Keras
* TensorFlow
* NumPy
* Matplotlib
* Pandas

### Installation
1. Clone the repository:
```
git clone https://github.com/your-username/deep-space-radio-signals-classification.git
```

2. Navigate to the project directory:
```
cd deep-space-radio-signals-classification
```

### Running the Project
1. Open the Jupyter notebook:
```
jupyter notebook
```
2. Open Project_Notebook_Main.ipynb file
3. Follow the instructions in the notebook to run the code cells and train the CNN model.


## Results
The project successfully builds and trains a CNN to classify deep space radio signals with a high level of accuracy. The model's performance and the classification results are visualized in the notebook.


## Conclusion
This project demonstrates the application of deep learning techniques to classify radio signals from space, showcasing the use of Keras and TensorFlow for image classification tasks. It provides a hands-on approach to understanding and implementing CNNs for practical data analysis problems.


## Contact
I'm always excited to connect with fellow data and machine learning enthusiasts, potential employers, and anyone interested in Data Science, Machine Learning and Artificial Intelligence. Feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/chintan-patel2251/).

Thank you for visiting my repository. I look forward to your feedback and the opportunity to contribute to impactful projects!
