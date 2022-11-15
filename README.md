# Smart-Meeting-Minutes
Creating a meeting transcriber using voice detection AI in CNN and CNN based on transfer learning

<h3>Introduction</h3>
This project is based on Audio Transcription. We are to build 2 different deep learning models and compare them 
based on their evaluations. This project falls under the category of machine learning. We have a goal, dataset, model, 
and evaluation as any machine learning problem. In our case, our goal is to build a model which recognises words 
from the audio input. A basic supervised learning model requires the dataset and its labels or supervisory signal. The 
dataset must be split into training and testing sets and fed to our neural network. Just like a regular model requires 
lots of training data to learn a specific class, we have training data for each word. Each word is a class. Multiple 
speakers speak the same word, which has their waveform; although different speakers, the waveform appears 
similar. We feed the neural network different waveforms of the same word to be able to recognise that word from a 
different speaker. Our dataset is in audio form with a folder of different words containing multiple speakers speaking 
out the word. Our program converts the audio to its respective spectrogram, which becomes our data to be trained 
and tested, i.e. the Xtrain and Xtest variables. Then we attach them with their respective labels, which are the words, 
i.e. the Ytrain and Ytest variables.


<h3>System Architecture</h3>
We are building two architectures: CNN and CNN, based on transfer learning

- CNN

For CNN, the name speaks for itself; a few layers of convolution and pooling layers are kept for feature learning and 
reduction, reducing the image’s complexity, learning only the essential features, and then predicting using a fully 
connected layer. We are using three layers of convolution layer and a pooling layer with a final dense layer.

- CNN via Transfer Learning


For the second model, CNN architecture was used in conjunction with a pre-trained model using the ImageNet 
database. The model consisted of one Baselayer followed by MaxPooling and Dropout layers, and finally, four Dense 
layers to learn all the features from the image. Softmax was used on the final activation layer. Adam was used as the 
optimiser, and categorical_crossentropy was used for the loss function. Early stopping was also implemented to 
prevent overfitting with the patience of 10 epochs.
