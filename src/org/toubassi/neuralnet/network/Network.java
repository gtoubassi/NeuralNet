package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.matrix.Matrix;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 */
public class Network {

    public static class TrainingPair {

        public TrainingPair(Matrix inputVector, Matrix outputVector) {
            this.inputVector = inputVector;
            this.outputVector = outputVector;
        }

        public Matrix inputVector;
        public Matrix outputVector;
    }

    private static Random random = new Random(12345L);

    private Matrix unitVectorOutputSize;
    private Matrix unitVectorHiddenLayerSize;

    public Matrix hiddenLayerWeights;
    public Matrix gHiddenLayerWeights;

    public Matrix hiddenLayerBiasesVector;
    public Matrix gHiddenLayerBiasesVector;

    public Matrix outputLayerWeights;
    public Matrix gOutputLayerWeights;

    public Matrix outputLayerBiasesVector;
    public Matrix gOutputLayerBiasesVector;

    public Network(int numInputs, int numHiddenNodes, int numOutputs) {
        hiddenLayerWeights = new Matrix(numHiddenNodes, numInputs);
        hiddenLayerBiasesVector = new Matrix(numHiddenNodes, 1);
        outputLayerWeights = new Matrix(numOutputs, hiddenLayerBiasesVector.getRows());
        outputLayerBiasesVector = new Matrix(numOutputs, 1);

        gHiddenLayerWeights = new Matrix(numHiddenNodes, numInputs);
        gHiddenLayerBiasesVector = new Matrix(numHiddenNodes, 1);
        gOutputLayerWeights = new Matrix(numOutputs, gHiddenLayerBiasesVector.getRows());
        gOutputLayerBiasesVector = new Matrix(numOutputs, 1);

        unitVectorOutputSize = new Matrix(numOutputs, 1, 1f);
        unitVectorHiddenLayerSize = new Matrix(numHiddenNodes, 1, 1f);

        randomize(hiddenLayerWeights);
        randomize(hiddenLayerBiasesVector);
        randomize(outputLayerWeights);
        randomize(outputLayerBiasesVector);
    }

    public void save(DataOutputStream out) throws IOException {
        // numInputs
        out.writeInt(hiddenLayerWeights.getCols());
        // numHiddenNodes
        out.writeInt(hiddenLayerWeights.getRows());
        // numOutputs
        out.writeInt(outputLayerWeights.getRows());

        hiddenLayerWeights.save(out);
        hiddenLayerBiasesVector.save(out);
        outputLayerWeights.save(out);
        outputLayerBiasesVector.save(out);
    }

    public static Network load(DataInputStream in) throws IOException {
        Network net = new Network(in.readInt(), in.readInt(), in.readInt());
        net.hiddenLayerWeights.load(in);
        net.hiddenLayerBiasesVector.load(in);
        net.outputLayerWeights.load(in);
        net.outputLayerBiasesVector.load(in);
        return net;
    }

    private void randomize(Matrix m) {
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                m.set(i, j, (float)random.nextGaussian());
            }
        }
    }

    public Matrix evaluate(Matrix inputVector) {
        return evaluate(inputVector, null);
    }

    public Matrix evaluate(Matrix inputVector, Matrix hiddenLayerOutputs) {
        Matrix hiddenLayerOutputVector = evaluateLayer(inputVector, hiddenLayerWeights, hiddenLayerBiasesVector);

        // To support "wire tapping" the hidden layer
        if (hiddenLayerOutputs != null) {
            hiddenLayerOutputs.setFrom(hiddenLayerOutputVector);
        }

        return evaluateLayer(hiddenLayerOutputVector, outputLayerWeights, outputLayerBiasesVector);
    }

    private static Matrix evaluateLayer(Matrix inputVector, Matrix weights, Matrix biasesVector) {
        Matrix WI = weights.times(inputVector);
        Matrix WI_plus_Biases = WI.plus(biasesVector);
        sigmoid(WI_plus_Biases);
        return WI_plus_Biases;
    }

    private static void sigmoid(Matrix m) {

        for (int i = 0, rowCount = m.getRows(); i < rowCount; i++) {
            for (int j = 0, colCount = m.getCols(); j < colCount; j++) {
                m.set(i, j, sigmoid(m.get(i, j)));
            }
        }
    }

    private static float sigmoid(float z) {
        return (float)(1.0/(1.0 + Math.exp(-z)));
    }

    public void train(List<TrainingPair> data, int miniBatchSize, float learningRate) {
        List<TrainingPair> shuffledData = new ArrayList<TrainingPair>(data);

        Collections.shuffle(shuffledData, random);

        for (int i = 0; i < shuffledData.size(); i += miniBatchSize) {
            trainBatch(shuffledData.subList(i, Math.min(shuffledData.size(), i + miniBatchSize)), learningRate);
        }
    }

    private void resetGradient() {
        gHiddenLayerWeights.setAll(0.0f);
        gHiddenLayerBiasesVector.setAll(0.0f);
        gOutputLayerWeights.setAll(0.0f);
        gOutputLayerBiasesVector.setAll(0.0f);
    }

    private void trainBatch(List<TrainingPair> data, float learningRate) {

        resetGradient();

        for (TrainingPair pair : data) {
            trainPair(pair);
        }

        // Average out the gradient and multiple by negative learning factor, and add to weights/biases;

        float factor = - learningRate / data.size();

        hiddenLayerWeights = hiddenLayerWeights.plus(gHiddenLayerWeights.scalarTimes(factor));
        hiddenLayerBiasesVector = hiddenLayerBiasesVector.plus(gHiddenLayerBiasesVector.scalarTimes(factor));
        outputLayerWeights = outputLayerWeights.plus(gOutputLayerWeights.scalarTimes(factor));
        outputLayerBiasesVector = outputLayerBiasesVector.plus(gOutputLayerBiasesVector.scalarTimes(factor));

        resetGradient();
    }

    private void trainPair(TrainingPair pair) {
        trainViaBackpropagation(pair.inputVector, pair.outputVector);
    }

    private void trainViaBackpropagation(Matrix inputVector, Matrix targetVector) {
        Matrix hiddenLayerOutputVector = evaluateLayer(inputVector, hiddenLayerWeights, hiddenLayerBiasesVector);
        Matrix outputVector = evaluateLayer(hiddenLayerOutputVector, outputLayerWeights, outputLayerBiasesVector);

        Matrix dE_db2 = outputVector.minus(targetVector).hadamardTimes(outputVector).hadamardTimes(unitVectorOutputSize.minus(outputVector));

        Matrix dE_dw2 = dE_db2.times(hiddenLayerOutputVector.transpose());

        Matrix dE_db1 = hiddenLayerOutputVector.hadamardTimes(unitVectorHiddenLayerSize.minus(hiddenLayerOutputVector)).hadamardTimes(outputLayerWeights.transpose().times(dE_db2));

        Matrix dE_dw1 = dE_db1.times(inputVector.transpose());

        gHiddenLayerWeights = gHiddenLayerWeights.plus(dE_dw1);
        gHiddenLayerBiasesVector = gHiddenLayerBiasesVector.plus(dE_db1);
        gOutputLayerWeights = gOutputLayerWeights.plus(dE_dw2);
        gOutputLayerBiasesVector = gOutputLayerBiasesVector.plus(dE_db2);
    }

    private float computeError(Matrix outputVector, Matrix targetVector) {
        return outputVector.minus(targetVector).arrayPow(2f).sum() / 2f;
    }

    public float evaluateRMS(List<TrainingPair> pairs) {
        float accumulatedRMS = 0f;

        for (TrainingPair pair : pairs) {
            Matrix output = evaluate(pair.inputVector);
            accumulatedRMS += output.rmsError(pair.outputVector);
        }
        return accumulatedRMS / pairs.size();
    }

    private NumberFormat format = new DecimalFormat(" #.#####;-#.#####");

    private void printRow(Matrix m, int i) {
        if (i >= m.getRows()) {
            System.out.print("     ");
            for (int j = 0, colCount = m.getCols(); j < colCount; j++) {
                System.out.print("      ");
            }
            return;
        }
        System.out.print("| ");
        for (int j = 0, colCount = m.getCols(); j < colCount; j++) {
            System.out.print(format.format(m.get(i, j)));
            System.out.print(" ");
        }
        System.out.print("|");
    }

    public void print(Matrix inputVector, Matrix targetVector) {
        int rowCount = hiddenLayerWeights.getRows();
        Matrix hiddenLayerOutputVector = evaluateLayer(inputVector, hiddenLayerWeights, hiddenLayerBiasesVector);
        Matrix outputVector = evaluateLayer(hiddenLayerOutputVector, outputLayerWeights, outputLayerBiasesVector);

        for (int i = 0; i < rowCount; i++) {
            printRow(hiddenLayerWeights, i);
            printRow(inputVector, i);
            System.out.print("  ");
            printRow(hiddenLayerBiasesVector, i);

            System.out.print("    ");
            printRow(hiddenLayerOutputVector, i);
            System.out.println();
        }

        System.out.println();

        rowCount = hiddenLayerOutputVector.getRows();
        for (int i = 0; i < rowCount; i++) {
            printRow(outputLayerWeights, i);
            printRow(hiddenLayerOutputVector, i);
            System.out.print("  ");
            printRow(outputLayerBiasesVector, i);

            System.out.print("    ");
            printRow(outputVector, i);
            System.out.print(" ");
            printRow(targetVector, i);
            System.out.println();
        }
    }
}
