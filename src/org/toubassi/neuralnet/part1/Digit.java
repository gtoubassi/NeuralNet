package org.toubassi.neuralnet.part1;

import org.toubassi.neuralnet.matrix.Matrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by gtoubassi on 9/13/15.
 */
public class Digit {

    private Matrix matrix;
    private int digit;

    public Digit(Matrix m, int digit) {
        this.matrix = m;
        this.digit = digit;
    }

    public Matrix getMatrix() {
        return matrix;
    }

    public int getDigit() {
        return digit;
    }

    public float getPixel(int x, int y) {
        return matrix.get(y * 28 + x, 0);
    }

    public static List<Digit> load(String images, String labels) throws IOException {
        List<Digit> digits = new ArrayList<>();
        BufferedImage image = ImageIO.read(new File(images));

        if (image.getWidth() % 28 != 0 || image.getHeight() % 28 != 0) {
            throw new IllegalArgumentException("Image must be a multiple of 28x28");
        }

        Reader reader = new BufferedReader(new FileReader(labels));

        for (int i = 0; i < image.getHeight() / 28; i++) {
            for (int j = 0; j < image.getWidth() / 28; j++) {

                Matrix matrix = new Matrix(28 * 28, 1);

                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        int pixel = image.getRGB(x + j * 28, y + i * 28);
                        float grayscale = 1f - (pixel & 0xff) / 255f;
                        matrix.set(y * 28 + x, 0, grayscale);
                    }
                }

                int digit = reader.read();
                if (digit == -1) {
                    throw new EOFException();
                }
                if (digit < '0' || digit > '9') {
                    throw new IllegalArgumentException("Unexpected character in label data: " + (char)digit);
                }
                digit = digit - '0';
                digits.add(new Digit(matrix, digit));
            }
        }

        reader.close();

        return digits;
    }

    public void dump() {
        System.out.println("Digit: " + digit);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                System.out.print(getPixel(x, y) > .5 ? '#' : ' ');
            }
            System.out.println();
        }
    }

    public static int convertOutputToDigit(Matrix output) {
        float maxValue = -1;
        int maxDigit = -1;
        for (int i = 0; i < 10; i++) {
            if (output.get(i, 0) > maxValue) {
                maxValue = output.get(i, 0);
                maxDigit = i;
            }
        }
        return maxDigit;
    }

    public static void main(String[] args) throws IOException {
        List<Digit> digits = Digit.load(args[0], args[1]);
        System.out.println(digits.size());

        for (int i = 0; i < 10; i++) {
            digits.get(i).dump();
        }
    }
}
