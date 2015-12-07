package org.toubassi.neuralnet.data;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 */
public class BitmapDigitLoader {
    public static List<Digit> load(String bitmapPath, String labelsPath) throws IOException {
        BufferedImage masterImage = ImageIO.read(new File(bitmapPath));

        if (masterImage.getWidth() % 28 != 0 || masterImage.getHeight() % 28 != 0) {
            throw new IllegalArgumentException(bitmapPath + " must have dimensions multiples of 28x28");
        }

        int numImagesHorizontal = masterImage.getWidth() / 28;
        int numImagesVertical = masterImage.getHeight() / 28;

        Reader reader = null;
        if (labelsPath != null) {
            reader = new BufferedReader(new FileReader(labelsPath));
        }

        List<Digit> digits = new ArrayList<>();
        for (int i = 0; i < numImagesVertical; i++) {
            for (int j = 0; j < numImagesHorizontal; j++) {
                Image image = new AWTImage(masterImage, j * 28, i * 28, 28, 28);
                int digit = -1;
                if (reader != null) {
                    int ch = reader.read();
                    if (ch == -1) {
                        throw new IOException("Unexpected EOF on " + labelsPath);
                    }
                    else if  (ch < '0' || ch > '9') {
                        throw new IOException("Unexpected character '" + ch + "' on " + labelsPath);
                    }
                    digit = ch - '0';
                }
                digits.add(new Digit(image, digit));
            }
        }

        if (reader != null) {
            reader.close();
        }

        return digits;
    }
}
