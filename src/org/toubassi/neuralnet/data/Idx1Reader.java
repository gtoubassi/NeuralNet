package org.toubassi.neuralnet.data;

import java.io.*;

/**
 * From: http://yann.lecun.com/exdb/mnist/
 *
 * [offset] [type]          [value]          [description]
 *  0000     32 bit integer  0x00000801(2049) magic number (MSB first)
 *  0004     32 bit integer  60000            number of items
 *  0008     unsigned byte   ??               label
 *  0009     unsigned byte   ??               label
 *  ........
 *  xxxx     unsigned byte   ??               label
 *
 *  The labels values are 0 to 9.
 */
public class Idx1Reader {
    private DataInputStream in;

    private int numLabels;

    public void open(String path) throws IOException {
        in = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));

        int magic = in.readInt();

        if (magic != 0x801) {
            throw new IOException("Expected magic number 0x00000801");
        }

        numLabels = in.readInt();
    }

    public int[] readLabels() throws IOException {
        int[] labels = new int[numLabels];

        for (int i = 0; i < labels.length; i++) {
            labels[i] = in.read();
            if (labels[i] == -1) {
                throw new EOFException();
            }
        }

        return labels;
    }

    public void close() throws IOException {
        in.close();
    }

    public static void main(String args[]) throws IOException {
        Idx1Reader reader = new Idx1Reader();
        reader.open(args[0]);

        int[] labels = reader.readLabels();

        for (int i = 0; i < 20; i++) {
            System.out.println(i + ": " + labels[i]);
        }

        reader.close();
    }
}
