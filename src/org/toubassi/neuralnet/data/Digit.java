package org.toubassi.neuralnet.data;

import org.toubassi.neuralnet.matrix.Matrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 */
public class Digit {
    private Image image;
    private int digit;
    private Matrix targetVector;

    public Digit(Image image, int digit) {
        this.image = image;
        this.digit = digit;

        targetVector = new Matrix(10, 1);
        if (digit >= 0) {
            targetVector.set(getDigit(), 0, 1);
        }
    }

    public Image getImage() {
        return image;
    }

    public int getDigit() {
        return digit;
    }

    public Matrix getInputVector() {
        return image.getMatrix();
    }


    public Matrix getTargetOutputVector() {
        return targetVector;
    }

    public static int convertOutputToDigit(Matrix outputVector) {
        int maxDigit = 0;
        float maxValue = Integer.MIN_VALUE;

        for (int i = 0; i < outputVector.getRows(); i++) {
            if (outputVector.get(i, 0) > maxValue) {
                maxValue = outputVector.get(i, 0);
                maxDigit = i;
            }
        }

        return maxDigit;
    }
}
