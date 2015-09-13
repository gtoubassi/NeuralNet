package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.data.BitmapDigitLoader;
import org.toubassi.neuralnet.data.Digit;
import org.toubassi.neuralnet.data.Image;
import org.toubassi.neuralnet.matrix.Matrix;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

/**
 */
public class DigitRecognizer {

    private static boolean verbose = true;

    public static void main(String[] args) throws IOException, InterruptedException {
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(args[1])));
        Network network = Network.load(in);
        in.close();

        List<Digit> digits = BitmapDigitLoader.load(args[0], null);

        for (Digit digit : digits) {
            Image image = digit.getImage();
            if (verbose) {
                image.print(System.out);
            }
            Matrix output = network.evaluate(image.getMatrix());
            Digit.DigitScore[] scores = Digit.getDigitsOrderedByScore(output);

            System.out.println(scores[0].digit + " (" + scores[0].score + ")  "
                    + "[ Others "
                    + scores[1].digit + " (" + scores[1].score + ")  "
                    + scores[2].digit + " (" + scores[2].score + ") ]");
            if (verbose) {
                System.out.println();
            }
        }

        if (!verbose) {
            System.out.println();
        }
    }
}
