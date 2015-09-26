package org.toubassi.neuralnet.part2;

import org.toubassi.neuralnet.part1.Matrix;
import org.toubassi.neuralnet.part1.Digit;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by gtoubassi on 9/17/15.
 */
public class Network {

    private Random random = new Random(12345L);

    private Matrix hiddenLayerWeights;
    private Matrix hiddenLayerBiases;
    private Matrix outputLayerWeights;
    private Matrix outputLayerBiases;

    public Network() {
        hiddenLayerWeights = new Matrix(30, 28*28);
        hiddenLayerBiases = new Matrix(30, 1);
        outputLayerWeights = new Matrix(10, 30);
        outputLayerBiases = new Matrix(10, 1);

    }

    private void initMatrix(Matrix m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                m.set(i, j, (float)random.nextGaussian());
            }
        }
    }

    public Matrix evaluate(Matrix input) {
        Matrix hiddenLayerOutput = evaluateLayer(input, hiddenLayerWeights, hiddenLayerBiases);
        return evaluateLayer(hiddenLayerOutput, outputLayerWeights, outputLayerBiases);
    }

    private Matrix evaluateLayer(Matrix input, Matrix weights, Matrix biases) {
        Matrix m = weights.times(input).plus(biases);
        sigmoidInPlace(m);
        return m;
    }

    private void sigmoidInPlace(Matrix m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                float v = m.get(i, j);
                v = 1f / (1f + (float)Math.exp(-v));
                m.set(i, j, v);
            }
        }
    }

    public static void main(String[] args) throws IOException {
        List<Digit> trainingDigits = Digit.load(args[0], args[1]);
        System.out.println("Training data set size: " + trainingDigits.size());
        List<Digit> testDigits = Digit.load(args[2], args[3]);
        System.out.println("Test data set size: " + testDigits.size());

        Network network = new Network();

        int numCorrect = 0;
        for (int i = 0; i < trainingDigits.size(); i++) {
            Digit digit = trainingDigits.get(i);
            Matrix output = network.evaluate(digit.getMatrix());
            int recognized = Digit.convertOutputToDigit(output);

            if (recognized == digit.getDigit()) {
                numCorrect++;
            }
        }

        System.out.println(numCorrect + " / " + trainingDigits.size() + "  (" + (numCorrect / ((float)trainingDigits.size())) + ")");
    }
}
