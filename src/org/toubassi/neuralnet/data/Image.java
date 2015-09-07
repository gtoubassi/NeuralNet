package org.toubassi.neuralnet.data;

import org.toubassi.neuralnet.matrix.Matrix;

import java.io.PrintStream;

/**
 */
public abstract class Image {
    private static String grayChars = " .:-=+*#%@@";

    public abstract int getWidth();

    public abstract int getHeight();

    public abstract void setPixel(int x, int y, float value);

    public abstract Matrix getMatrix();

    public abstract float getPixel(int x, int y);

    public void print(PrintStream out) {
        for (int y = 0, height = getHeight(); y < height; y++) {
            for (int x = 0, width = getWidth(); x < width; x++) {
                char ch = grayChars.charAt((int) (getPixel(x, y) * 10));
                out.print(ch);
            }
            out.println();
        }
    }
}
