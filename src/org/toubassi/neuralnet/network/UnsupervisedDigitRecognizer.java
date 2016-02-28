package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.data.BitmapDigitLoader;
import org.toubassi.neuralnet.data.Digit;
import org.toubassi.neuralnet.matrix.Matrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by gtoubassi on 2/19/16.
 */
public class UnsupervisedDigitRecognizer {

    public static void main(String[] args) throws IOException {
        trainUnsupervisedNetwork(args);
        analyzeUnsupervisedNetwork(args);
    }

    private static void writeDigitsImage(List<Digit> digits, String filename) throws IOException {
        int masterImageWidth = (int)Math.sqrt(digits.size());
        int masterImageHeight = digits.size() / masterImageWidth + 1;
        BufferedImage masterImage = new BufferedImage(28 * masterImageWidth, 28 * masterImageHeight, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D graphics = masterImage.createGraphics();
        AffineTransformOp op = new AffineTransformOp(new AffineTransform(), AffineTransformOp.TYPE_BICUBIC);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, masterImage.getWidth(), masterImage.getHeight());

        for (int i = 0; i < digits.size(); i++) {
            Digit digit = digits.get(i);
            Matrix vector = digit.getInputVector();
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    float gray = 1f - vector.get(y * 28 + x, 0);
                    graphics.setColor(new Color(gray, gray, gray));
                    graphics.drawRect(x + 28 * (i % masterImageWidth), y + 28 * (i / masterImageWidth), 1, 1);
                }
            }
        }

        graphics.dispose();
        ImageIO.write(masterImage, "png", new File(filename));
    }

    private static void analyzeUnsupervisedNetwork(String[] args) throws IOException {
        System.out.print("Loading data...");
        List<Digit> testDigits = BitmapDigitLoader.load(args[2], args[3]);
        List<Network.TrainingPair> trainingDataset = new ArrayList<>();
        List<Network.TrainingPair> testDataset = new ArrayList<>();

        for (Digit digit : testDigits) {
            testDataset.add(new Network.TrainingPair(digit.getInputVector(), digit.getInputVector()));
        }

        System.out.println("Done");

        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream("data/unsupervised_network_7")));
        Network network = Network.load(in);
        in.close();

        List<Digit>[] histogram = new ArrayList[(1 << network.getNumHiddenNodes()) * 10];
        for (int i = 0; i < histogram.length; i++) {
            histogram[i] = new ArrayList();
        }

        Matrix hiddenOutputs = new Matrix(network.getNumHiddenNodes(), 1);
        for (Digit digit : testDigits) {
            network.evaluate(digit.getInputVector(), hiddenOutputs);
            int classification = 0;
            for (int row = 0; row < hiddenOutputs.getRows(); row++) {
                classification = (classification << 1) | (hiddenOutputs.get(row, 0) > .8 ? 1 : 0);
            }
            histogram[classification * 10 + digit.getDigit()].add(digit);
        }

        for (int i = 0; i < (1 << network.getNumHiddenNodes()); i++) {
            int sum = 0;
            List<Digit> digits = new ArrayList<Digit>();
            for (int j = 0; j < 10; j++) {
                sum += histogram[i * 10 + j].size();
                digits.addAll(histogram[i * 10 + j]);
            }
            if (sum > 50) {
                System.out.print(i + "\t" + sum);
                for (int j = 0; j < 10; j++) {
                    System.out.print("\t" + histogram[i * 10 + j].size() * 100 / sum + "%");
                }
                System.out.println();
                Collections.shuffle(digits);
                writeDigitsImage(digits, "/tmp/category_" + i + ".png");
            }
        }
/**/
        /*
        BufferedImage masterImage = new BufferedImage(28 * 20, 28 * 10, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D graphics = masterImage.createGraphics();
        AffineTransformOp op = new AffineTransformOp(new AffineTransform(), AffineTransformOp.TYPE_BICUBIC);
        graphics.setColor(Color.WHITE);
        graphics.fillRect(0, 0, masterImage.getWidth(), masterImage.getHeight());

        for (int i = 0; i < 100; i++) {
            Digit digit = testDigits.get(i);
            Matrix vector = digit.getInputVector();
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    float gray = 1f - vector.get(y * 28 + x, 0);
                    graphics.setColor(new Color(gray, gray, gray));
                    graphics.drawRect(x + 28 * (i / 10), y + 28 * (i % 10), 1, 1);
                }
            }
            vector = network.evaluate(digit.getInputVector());
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    float gray = 1f - vector.get(y * 28 + x, 0);
                    graphics.setColor(new Color(gray, gray, gray));
                    graphics.drawRect(280 + x + 28 * (i / 10), y + 28 * (i % 10), 1, 1);
                }
            }
            //graphics.drawImage(bufferedImage, op, 28 * (i / 10), 28 * (i % 10));
        }

        graphics.dispose();
        ImageIO.write(masterImage, "png", new File("/tmp/out.png"));

        //DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[4])));
        //network.save(out);
        //out.close();
    */
    }

    private static void trainUnsupervisedNetwork(String[] args) throws IOException {
        System.out.print("Loading data...");
        List<Digit> trainDigits = BitmapDigitLoader.load(args[0], args[1]);
        List<Digit> testDigits = BitmapDigitLoader.load(args[2], args[3]);
        List<Network.TrainingPair> trainingDataset = new ArrayList<>();
        List<Network.TrainingPair> testDataset = new ArrayList<>();

        for (Digit digit : trainDigits) {
            trainingDataset.add(new Network.TrainingPair(digit.getInputVector(), digit.getInputVector()));
        }
        for (Digit digit : testDigits) {
            testDataset.add(new Network.TrainingPair(digit.getInputVector(), digit.getInputVector()));
        }

        System.out.println("Done");

        int[] hiddenNodes = {4};
        int[] miniBatchSize = {25, 50, 75};
        float[] learningRate = {.1f, .2f, .05f};

        for (int i = 0; i < hiddenNodes.length; i++) {
            for (int j = 0; j < miniBatchSize.length; j++) {
                for (int k = 0; k < learningRate.length; k++) {
                    Network network = evalNetwork(2000, hiddenNodes[i], miniBatchSize[j], learningRate[k], trainingDataset, testDataset);
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[4] + "_" + i + "_" + j + "_" + k)));
                    network.save(out);
                    out.close();
                }
            }
        }
    }

    private static Network evalNetwork(int numEpochs, int numHiddenNodes, int miniBatchSize, float learningRate, List<Network.TrainingPair> trainingDataset, List<Network.TrainingPair> testDataset) {
        Network network = new Network(28 * 28, numHiddenNodes, 28 * 28);

        float rms;
        rms = network.evaluateRMS(testDataset);
        System.out.println("Test RMS error " + rms);

        for (int i = 0; i < numEpochs; i++) {
            System.out.print("Training (iteration " + (i + 1) + " of " + (numEpochs) + ")... ");
            network.train(trainingDataset, miniBatchSize, learningRate);
            System.out.println("Done");
            if (i % 5 == 0) {
                rms = network.evaluateRMS(testDataset);
                System.out.println("RMS error " + rms + "  " + numHiddenNodes + " " + miniBatchSize + " " + learningRate);
            }
        }
        return network;
    }
}
