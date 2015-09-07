package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.data.BitmapDigitLoader;
import org.toubassi.neuralnet.data.Digit;
import org.toubassi.neuralnet.data.MNISTDigitLoader;
import org.toubassi.neuralnet.matrix.Matrix;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Usage:
 * DigitTrainer train_digits.png train_digits.txt test_digits.png test_digits.txt output_network
 */
public class DigitTrainer {

    public static void evalDigits(Network network, List<Digit> digits) {
        float accumulatedRMS = 0f;
        int correct = 0;
        for (Digit digit : digits) {
            Matrix output = network.evaluate(digit.getInputVector());
            accumulatedRMS += output.rmsError(digit.getTargetOutputVector());
            int computed = digit.convertOutputToDigit(output);
            if (computed == digit.getDigit()) {
                correct++;
            }
        }
        System.out.println("rms = " + accumulatedRMS + " correct = " + ((float) correct) / digits.size());
    }

    public static void main(String[] args) throws IOException {
        System.out.print("Loading data...");
        List<Digit> trainDigits = BitmapDigitLoader.load(args[0], args[1]);
        List<Digit> testDigits = BitmapDigitLoader.load(args[2], args[3]);
        List<Network.TrainingPair> dataset = new ArrayList<>();

        for (Digit digit : trainDigits) {
            dataset.add(new Network.TrainingPair(digit.getInputVector(), digit.getTargetOutputVector()));
        }

        System.out.println("Done");

        System.out.println(trainDigits.size() + " training samples, " + testDigits.size() + " test samples");

        Network network = new Network(28 * 28, 30, 10);
        evalDigits(network, testDigits);

        int numEpochs = 30;
        for (int i = 0; i < numEpochs; i++) {
            System.out.print("Training (iteration " + (i + 1) + " of " + (numEpochs) + ") ... ");
            network.train(dataset, 10, 3f);
            System.out.println("Done");
            evalDigits(network, testDigits);
        }

        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[4])));
        network.save(out);
        out.close();
    }
}
