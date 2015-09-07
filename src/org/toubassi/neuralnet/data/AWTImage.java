package org.toubassi.neuralnet.data;

import org.toubassi.neuralnet.matrix.Matrix;

import java.awt.image.BufferedImage;

/**
 */
public class AWTImage extends Image {

    private Matrix m;
    private int xOffset;
    private int yOffset;
    private int height;
    private int width;

    public AWTImage(BufferedImage image, int xOffset, int yOffset, int width, int height) {
        this.xOffset = xOffset;
        this.yOffset = yOffset;
        this.height = height;
        this.width = width;
        m = new Matrix(width * height, 1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(xOffset + x, yOffset + y);
                float r = ((rgb >> 16) & 0xff) / 255f;
                float g = ((rgb >> 8) & 0xff) / 255f;
                float b = (rgb & 0xff) / 255f;
                float gray = 1 - Math.min(1.0f, 0.2989f * r + 0.5870f * g + 0.1140f * b);
                m.set(y * height + x, 0, gray);
            }
        }
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
        throw new UnsupportedOperationException("Can't set pixels on AWTImage");
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
