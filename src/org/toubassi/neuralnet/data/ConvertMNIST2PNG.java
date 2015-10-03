package org.toubassi.neuralnet.data;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 */
public class ConvertMNIST2PNG {

    public static void main(String[] args) throws IOException {
        List<Digit> digits = MNISTDigitLoader.load(args[0], args[1]);
        // We lose a few here.
        int imageRows = (int)Math.sqrt(digits.size());

        while (digits.size() % imageRows != 0) {
            imageRows--;
        }

        int imageCols = digits.size() / imageRows;

        BufferedImage image = new BufferedImage(28 * imageCols, 28 * imageRows, BufferedImage.TYPE_3BYTE_BGR);
        FileWriter writer = new FileWriter("/tmp/digits.txt");

        for (int row = 0; row < imageRows; row++) {
            for (int col = 0; col < imageCols; col++) {
                Digit digit = digits.get(row * imageCols + col);
                Image digitImage = digit.getImage();
                for (int x = 0; x < 28; x++) {
                    for (int y = 0; y < 28; y++) {
                        float pixel = 1f - Math.min(1.0f, Math.max(0, digitImage.getPixel(x, y)));
                        int pixelAsByte = ((int) (pixel * 255)) & 0xff;
                        int bgr = (pixelAsByte << 16) | (pixelAsByte << 8) | pixelAsByte;
                        image.setRGB(x + 28 * col, y + 28 * row, bgr);
                    }
                }
                writer.append((char)('0' + digit.getDigit()));
            }
            writer.append('\n');
        }

        ImageIO.write(image, "png", new File("/tmp/digits.png"));
        writer.close();

    }
}
