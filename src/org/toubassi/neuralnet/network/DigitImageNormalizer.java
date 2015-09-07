package org.toubassi.neuralnet.network;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A hacky image normalizer.  Write some digits in *one* a horizontal line
 * on a white piece of paper, take a picture, make sure it is just text
 * on a white background (paper needs to be "full bleed").  Run it through
 * this and it will output the same digits normalized into a row of 28x28
 * images suitable for running against the DigitRecognizer.  The means of
 * normalization is pretty hacky and is based on exactly one sample of me
 * writing 0123456789.  It assumes the imagemagick "convert" command is
 * installed and on the path.  It does try to achieve the same
 * normalization as the MNIST dataset as described in
 * http://yann.lecun.com/exdb/mnist/ and quoted below.
 *
 * "The original black and white (bilevel) images from NIST were size
 * normalized to fit in a 20x20 pixel box while preserving their aspect
 * ratio. The resulting images contain grey levels as a result of the
 * anti-aliasing technique used by the normalization algorithm. the
 * images were centered in a 28x28 image by computing the center of mass
 * of the pixels, and translating the image so as to position this point
 * at the center of the 28x28 field."
 */
public class DigitImageNormalizer {

    private static boolean isVerticalLineWhite(BufferedImage image, int x) {
        int height = image.getHeight();
        for (int y = 0; y < height; y++) {
            int rgb = image.getRGB(x, y) & 0xffffff;
            if (rgb != 0xffffff) {
                return false;
            }
        }
        return true;
    }

    private static boolean isHorizonalLineWhite(BufferedImage image, int y, int xOffset, int width) {
        for (int x = xOffset; x < xOffset + width; x++) {
            int rgb = image.getRGB(x, y) & 0xffffff;
            if (rgb != 0xffffff) {
                return false;
            }
        }
        return true;
    }

    public static List<BufferedImage> chopDigits(BufferedImage image) throws IOException, InterruptedException {
        List<BufferedImage> images = new ArrayList<>();

        for (int x = 0; x < image.getWidth(); x++) {

            for (; x < image.getWidth(); x++) {
                if (!isVerticalLineWhite(image, x)) {
                    break;
                }
            }

            if (x == image.getWidth()) {
                break;
            }

            int endX;
            for (endX = x + 1; endX < image.getWidth() - 1; endX++) {
                if (isVerticalLineWhite(image, endX)) {
                    break;
                }
            }

            int subImageWidth = endX - x;

            int y = 0;
            for (; y < image.getHeight(); y++) {
                if (!isHorizonalLineWhite(image, y, x, subImageWidth)) {
                    break;
                }
            }

            int subImageHeight = image.getHeight();
            while (isHorizonalLineWhite(image, subImageHeight - 1, x, subImageWidth)) {
                subImageHeight--;
            }

            BufferedImage subImage = image.getSubimage(x, y, subImageWidth, subImageHeight - y);

            ImageIO.write(subImage, "png", new File("/tmp/out.png"));
            Process process = Runtime.getRuntime().exec("convert /tmp/out.png -resize 20x20 /tmp/out2.png");
            process.waitFor();

            BufferedImage image20x20 = ImageIO.read(new File("/tmp/out2.png"));
            float centroidX = 0, centroidY = 0;
            int count = 0;
            for (int i = 0; i < image20x20.getWidth(); i++) {
                for (int j = 0; j < image20x20.getHeight(); j++) {
                    int rgb = image20x20.getRGB(i, j);
                    if (rgb != 0xffffff) {
                        centroidX += i;
                        centroidY += j;
                        count++;
                    }
                }
            }

            centroidX /= count;
            centroidY /= count;

            // Create a white 28x28 image
            BufferedImage image28x28 = new BufferedImage(28, 28, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D graphics = image28x28.createGraphics();
            graphics.fillRect(0, 0, 28, 28);
            graphics.drawImage(image20x20,
                    new AffineTransformOp(new AffineTransform(), AffineTransformOp.TYPE_BICUBIC),
                    Math.round(28/2 - centroidX), Math.round(28/2 - centroidY));
            graphics.dispose();

            images.add(image28x28);

            x = endX;
        }

        return images;
    }

    public static void main(String args[]) throws IOException, InterruptedException {
        Process process = Runtime.getRuntime().exec("convert " + args[0] + " -level 80%,80% /tmp/foo.png");
        process.waitFor();

        BufferedImage masterImage = ImageIO.read(new File("/tmp/foo.png"));

        List<BufferedImage> subImages = chopDigits(masterImage);

        BufferedImage normalizedImage = new BufferedImage(28 * subImages.size(), 28, BufferedImage.TYPE_3BYTE_BGR);
        Graphics2D graphics = normalizedImage.createGraphics();
        AffineTransformOp op = new AffineTransformOp(new AffineTransform(), AffineTransformOp.TYPE_BICUBIC);

        for (int i = 0; i < subImages.size(); i++) {
            graphics.drawImage(subImages.get(i), op, i * 28, 0);
        }

        graphics.dispose();

        ImageIO.write(normalizedImage, "png", new File(args[1]));
    }
}
