package org.toubassi.neuralnet.part4;

import org.toubassi.neuralnet.matrix.Matrix;
import org.toubassi.neuralnet.part3.Digit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by gtoubassi on 9/17/15.
 */
public class Network {

    private static class TrainingPair {
        Matrix input;
        Matrix output;

        public TrainingPair(Matrix input, Matrix output) {
          this.input = input;
          this.output = output;
        }
    }

    private static Random random = new Random(12345L);

    private Matrix hiddenLayerWeights;
    private Matrix hiddenLayerBiases;
    private Matrix outputLayerWeights;
    private Matrix outputLayerBiases;

    public Network() {
        hiddenLayerWeights = new Matrix(30, 28*28);
        hiddenLayerBiases = new Matrix(30, 1);
        outputLayerWeights = new Matrix(10, 30);
        outputLayerBiases = new Matrix(10, 1);

        initMatrix(hiddenLayerWeights);
        initMatrix(hiddenLayerBiases);
        initMatrix(outputLayerWeights);
        initMatrix(outputLayerBiases);
    }

    private void initMatrix(Matrix m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                m.set(i, j, (float)random.nextGaussian());
            }
        }
    }

    public Network clone() {
        Network n = new Network();
        n.hiddenLayerWeights = hiddenLayerWeights.clone();
        n.hiddenLayerBiases = hiddenLayerBiases.clone();
        n.outputLayerWeights = outputLayerWeights.clone();
        n.outputLayerBiases = outputLayerBiases.clone();
        return n;
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

    private void evaluateGradient() {
    }

    private static Matrix UnitVector10 = new Matrix(10,1, 1f);
    private static Matrix UnitVector30 = new Matrix(30,1, 1f);

    public void evalGradient(Matrix hiddenLayerWeightsGradient,
                             Matrix hiddenLayerBiasesGradient,
                             Matrix outputLayerWeightsGradient,
                             Matrix outputLayerBiasesGradient,
                             Matrix input,
                             Matrix targetOutput) {

        Matrix hiddenLayerOutput = evaluateLayer(input, hiddenLayerWeights, hiddenLayerBiases);
        Matrix output = evaluateLayer(hiddenLayerOutput, outputLayerWeights, outputLayerBiases);

        Matrix dE_db2 = output.minus(targetOutput).hadamardTimesInPlace(output).hadamardTimesInPlace(UnitVector10.minus(output));
        outputLayerBiasesGradient.setFrom(dE_db2);

        Matrix dE_dw2 = dE_db2.times(hiddenLayerOutput.transpose());
        outputLayerWeightsGradient.setFrom(dE_dw2);

        Matrix dE_db1 = hiddenLayerOutput.hadamardTimes(UnitVector30.minus(hiddenLayerOutput)).hadamardTimesInPlace(outputLayerWeights.transpose().times(dE_db2));
        hiddenLayerBiasesGradient.setFrom(dE_db1);

        Matrix dE_dw1 = dE_db1.times(input.transpose());
        hiddenLayerWeightsGradient.setFrom(dE_dw1);
    }


    public void trainOneEpoch(List<TrainingPair> trainingData, int batchSize, float learningRate) {
        Collections.shuffle(trainingData);

        Matrix hiddenLayerWeightsGradient = new Matrix(30, 28 * 28);
        Matrix hiddenLayerBiasesGradient = new Matrix(30, 1);
        Matrix outputLayerWeightsGradient = new Matrix(10, 30);
        Matrix outputLayerBiasesGradient = new Matrix(10, 1);

        Matrix hiddenLayerWeightsGradientSum = new Matrix(30, 28 * 28);
        Matrix hiddenLayerBiasesGradientSum = new Matrix(30, 1);
        Matrix outputLayerWeightsGradientSum = new Matrix(10, 30);
        Matrix outputLayerBiasesGradientSum = new Matrix(10, 1);


        for (int i = 0; i < trainingData.size(); i += batchSize) {
            List<TrainingPair> batch = trainingData.subList(i, Math.min(trainingData.size(), i + batchSize));

            hiddenLayerWeightsGradientSum.setAll(0f);
            hiddenLayerBiasesGradientSum.setAll(0f);
            outputLayerWeightsGradientSum.setAll(0f);
            outputLayerBiasesGradientSum.setAll(0f);

            for (TrainingPair pair : batch) {
                hiddenLayerWeightsGradient.setAll(0f);
                hiddenLayerBiasesGradient.setAll(0f);
                outputLayerWeightsGradient.setAll(0f);
                outputLayerBiasesGradient.setAll(0f);

                evalGradient(hiddenLayerWeightsGradient, hiddenLayerBiasesGradient, outputLayerWeightsGradient,
                        outputLayerBiasesGradient, pair.input, pair.output);

                hiddenLayerWeightsGradientSum.plusInPlace(hiddenLayerWeightsGradient);
                hiddenLayerBiasesGradientSum.plusInPlace(hiddenLayerBiasesGradient);
                outputLayerWeightsGradientSum.plusInPlace(outputLayerWeightsGradient);
                outputLayerBiasesGradientSum.plusInPlace(outputLayerBiasesGradient);
            }

            hiddenLayerWeightsGradientSum.scalarTimesInPlace(-learningRate / batch.size());
            hiddenLayerBiasesGradientSum.scalarTimesInPlace(-learningRate / batch.size());
            outputLayerWeightsGradientSum.scalarTimesInPlace(-learningRate / batch.size());
            outputLayerBiasesGradientSum.scalarTimesInPlace(-learningRate / batch.size());

            hiddenLayerWeights.plusInPlace(hiddenLayerWeightsGradientSum);
            hiddenLayerBiases.plusInPlace(hiddenLayerBiasesGradientSum);
            outputLayerWeights.plusInPlace(outputLayerWeightsGradientSum);
            outputLayerBiases.plusInPlace(outputLayerBiasesGradientSum);
        }
    }


    public float distance(Network n) {
        float sum = outputLayerBiases.minus(n.outputLayerBiases).arrayPow(2.0f).sum() +
                outputLayerWeights.minus(n.outputLayerWeights).arrayPow(2.0f).sum() +
                hiddenLayerBiases.minus(n.hiddenLayerBiases).arrayPow(2.0f).sum() +
                hiddenLayerWeights.minus(n.hiddenLayerWeights).arrayPow(2.0f).sum();

        return (float)Math.sqrt(sum);
    }

    public static void main(String[] args) throws IOException {
        List<Digit> trainingDigits = Digit.load(args[0], args[1]);
        System.out.println("Training data set size: " + trainingDigits.size());
        List<Digit> testDigits = Digit.load(args[2], args[3]);
        System.out.println("Test data set size: " + testDigits.size());

        List<TrainingPair> trainingData = new ArrayList<>();

        for (Digit digit : trainingDigits) {
            trainingData.add(new TrainingPair(digit.getMatrix(), digit.getOutputMatrix()));
        }

        List<Network> networks = new ArrayList<>();

        for (int k = 0; k < 1; k++) {

            Network network = new Network();

            Network last;

            for (int epoch = 0; epoch < 30; epoch++) {
                System.out.print("Epoch " + (epoch + 1) + "... ");
                last = network.clone();
                network.trainOneEpoch(trainingData, 10, 3f);
                System.out.println("Distance from last: " + network.distance(last));

                int numCorrect = 0;
                for (int i = 0; i < testDigits.size(); i++) {
                    Digit digit = testDigits.get(i);
                    Matrix output = network.evaluate(digit.getMatrix());
                    int recognized = Digit.convertOutputToDigit(output);

                    if (recognized == digit.getDigit()) {
                        numCorrect++;
                    }
                }

                System.out.println(numCorrect + " / " + testDigits.size() + "  (" + (numCorrect / ((float) testDigits.size())) + ")");
            }

            networks.add(network);

            if (networks.size() > 1) {
                System.out.println("Distance matrix:");
                for (Network n1 : networks) {
                    for (Network n2 : networks) {
                        System.out.print(n1.distance(n2) + ", ");
                    }
                    System.out.println();
                }
            }
        }
    }
}
