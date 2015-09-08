package org.toubassi.neuralnet.data;

import org.toubassi.neuralnet.matrix.Matrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
        return getDigitsOrderedByScore(outputVector)[0].digit;
    }

    public static DigitScore[] getDigitsOrderedByScore(Matrix outputVector) {
        int maxDigit = 0;
        float maxValue = Integer.MIN_VALUE;
        DigitScore[] scores = new DigitScore[10];

        for (int i = 0; i < outputVector.getRows(); i++) {
            scores[i] = new DigitScore(i, outputVector.get(i, 0));
        }

        Arrays.sort(scores);
        return scores;
    }

    public static class DigitScore implements Comparable<DigitScore> {
        public float score;
        public int digit;

        public DigitScore(int digit, float score) {
            this.digit = digit;
            this.score = score;
        }

        public int compareTo(DigitScore other) {
            if (other.score == score) {
                return 0;
            }
            return other.score - score < 0 ? -1 : 1;
        }
    }
}
