# NeuralNet

This is a java implementation of the simple handwriting recognition neural net outlined in the first two chapters of http://neuralnetworksanddeeplearning.com/chap1.html.

Derivation of the backpropagation algorithm was based on https://www.youtube.com/watch?v=aVId8KMsdUU (which has a few errors so view the actual slides at http://db.tt/yq6X4bQS)


### Get and Build

    % git clone https://github.com/gtoubassi/NeuralNet.git
    % cd NeuralNet/src
    % javac -classpath ../lib/junit-4.12.jar org/toubassi/neuralnet/*/*.java

### Train the neural net

The train and test data sets are based on the MNIST data referenced in neuralnetworksanddeeplearning.com.  The test data is used to evaluate the quality of the network, and is based on samples from subjects who DID NOT contribute to the training data.  After 30 iterations the accuracy should be ~95%.

    % java org.toubassi.neuralnet.network.DigitTrainer ../data/train_digits.png  ../data/train_digits.txt ../data/test_digits.png ../data/test_digits.txt ../data/network

### Normalize a handwriting sample

Write a set of digits in a single line on a white piece of paper, and take a photo.  Make sure the white paper extends the full range of the photo.  The normalization is based on what is described as the technique used to normalize the original MNIST data (http://yann.lecun.com/exdb/mnist).

    % java org.toubassi.neuralnet.network.DigitImageNormalizer ../data/gt_sample.png ../data/gt_sample_normalized.png

### Recognize the Sample

This outputs 0123956789 so it misses the 4 (90%).

    % java org.toubassi.neuralnet.network.DigitRecognizer ../data/gt_sample_normalized.png ../data/network
