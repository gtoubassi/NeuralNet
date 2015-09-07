package org.toubassi.neuralnet.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

/**
 *
 * From: http://yann.lecun.com/exdb/mnist/
 *
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000803(2051) magic number
 * 0004     32 bit integer  60000            number of images
 * 0008     32 bit integer  28               number of rows
 * 0012     32 bit integer  28               number of columns
 * 0016     unsigned byte   ??               pixel
 * 0017     unsigned byte   ??               pixel
 * ........
 * xxxx     unsigned byte   ??               pixel
 *
 * Pixels are organized row-wise. Pixel values are 0 to 255.
 * 0 means background (white), 255 means foreground (black).
 */
public class Idx3Reader {
    private DataInputStream in;

    private int numImages;
    private int rows;
    private int cols;

    public void open(String path) throws IOException {
        in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));

        int magic = in.readInt();

        if (magic != 0x803) {
            throw new IOException("Expected magic number 0x00000803");
        }

        numImages = in.readInt();
        rows = in.readInt();
        cols = in.readInt();
    }

    public int getNumImages() {
        return numImages;
    }

    public MatrixImage readImage() throws IOException {
        MatrixImage image = new MatrixImage(cols, rows);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int pixel = in.read();
                if (pixel == -1) {
                    if (r == 0 && c == 0) {
                        return null;
                    }
                    else {
                        throw new IOException("Partial image at end of file");
                    }
                }
                image.setPixel(c, r, pixel/255f);
            }
        }

        return image;
    }

    public void close() throws IOException {
        in.close();
    }

    public static void main(String args[]) throws IOException {
        Idx3Reader reader = new Idx3Reader();
        reader.open(args[0]);

        Image image;
        int count = 0;
        while ((image = reader.readImage()) != null) {
            System.out.println("\nImage number " + count);
            image.print(System.out);
            count++;

            if (count > 10) {
                break;
            }
        }

        reader.close();
    }
}
