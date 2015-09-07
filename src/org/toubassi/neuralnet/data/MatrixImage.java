package org.toubassi.neuralnet.data;

import org.toubassi.neuralnet.matrix.Matrix;

/**
 */
public class MatrixImage extends Image {

    private Matrix m;
    private int height;
    private int width;

    public MatrixImage() {
    }

    public MatrixImage(int width, int height) {
        this.height = height;
        this.width = width;
        m = new Matrix(width * height, 1);
    }

    @Override
    public int getWidth() {
        return width;
    }

    @Override
    public int getHeight() {
        return height;
    }

    @Override
    public void setPixel(int x, int y, float value) {
        m.set(y * height + x, 0, value);
    }

    @Override
    public Matrix getMatrix() {
        return m;
    }

    @Override
    public float getPixel(int x, int y) {
        return m.get(y * height + x, 0);
    }

}
