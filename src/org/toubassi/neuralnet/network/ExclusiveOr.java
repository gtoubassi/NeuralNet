package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.matrix.Matrix;
import org.toubassi.neuralnet.network.Network.TrainingPair;

import java.util.ArrayList;
import java.util.List;

/**
 */
public class ExclusiveOr {

    private static void main(String args[]) {
        Network network = new Network(2, 2, 1);
        List<TrainingPair> dataset = new ArrayList<TrainingPair>();

        Matrix in;
        Matrix out;

        in = new Matrix(2, 1);
        out = new Matrix(1, 1);
        in.set(0, 0, 0f);
        in.set(1, 0, 0f);
        out.set(0, 0, 0f);
        dataset.add(new TrainingPair(in, out));

        in = new Matrix(2, 1);
        out = new Matrix(1, 1);
        in.set(0, 0, 0f);
        in.set(1, 0, 1f);
        out.set(0, 0, 1f);
        dataset.add(new TrainingPair(in, out));

        in = new Matrix(2, 1);
        out = new Matrix(1, 1);
        in.set(0, 0, 1f);
        in.set(1, 0, 0f);
        out.set(0, 0, 1f);
        dataset.add(new TrainingPair(in, out));

        in = new Matrix(2, 1);
        out = new Matrix(1, 1);
        in.set(0, 0, 1f);
        in.set(1, 0, 1f);
        out.set(0, 0, 0f);
        dataset.add(new TrainingPair(in, out));

        network.print(dataset.get(0).inputVector, dataset.get(0).outputVector);

        System.out.println(network.evaluateRMS(dataset));

        for (int i = 0; i < 1000000; i++) {
            System.out.print("Training...");
            network.train(dataset, dataset.size(), .2f);
            System.out.println("Done");
            System.out.println(network.evaluateRMS(dataset));
        }

        System.out.println("Done");
    }

}
