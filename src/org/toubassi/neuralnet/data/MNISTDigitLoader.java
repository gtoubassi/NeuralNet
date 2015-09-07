package org.toubassi.neuralnet.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 */
public class MNISTDigitLoader {
    public static List<Digit> load(String digitsPath, String labelsPath) throws IOException {
        Idx3Reader imageReader = new Idx3Reader();
        imageReader.open(digitsPath);
        MatrixImage[] images = new MatrixImage[imageReader.getNumImages()];
        for (int i = 0; i < images.length; i++) {
            images[i] = imageReader.readImage();
        }
        imageReader.close();

        Idx1Reader labelReader = new Idx1Reader();
        labelReader.open(labelsPath);
        int[] labels = labelReader.readLabels();
        labelReader.close();

        if (labels.length != images.length) {
            throw new IOException("Size mismatch between digits (" + images.length + ") and labels (" + labels.length + ")" );
        }

        List<Digit> digits = new ArrayList<Digit>();

        for (int i = 0; i < images.length; i++) {
            digits.add(new Digit(images[i], labels[i]));
        }

        return digits;
    }
}
