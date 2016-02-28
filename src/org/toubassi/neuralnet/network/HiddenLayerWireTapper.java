package org.toubassi.neuralnet.network;

import org.toubassi.neuralnet.data.AWTImage;
import org.toubassi.neuralnet.data.BitmapDigitLoader;
import org.toubassi.neuralnet.data.Digit;
import org.toubassi.neuralnet.matrix.Matrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Created by gtoubassi on 9/7/15.
 */
public class HiddenLayerWireTapper {

    private static class Score implements Comparable<Score> {
        public float score;
        public Digit digit;

        public Score(Digit digit, float score) {
            this.digit = digit;
            this.score = score;
        }

        public int compareTo(Score other) {
            if (other.score == score) {
                return 0;
            }
            return other.score - score < 0 ? -1 : 1;
        }

    }
    public static void main(String[] args) throws IOException {
        List<Digit> digits = BitmapDigitLoader.load(args[0], null);
        DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(args[1])));
        Network network = Network.load(in);
        in.close();

        List<PriorityQueue<Score>> queues = new ArrayList<>();

        for (int i = 0; i < network.getNumHiddenNodes(); i++) {
            queues.add(new PriorityQueue<Score>());
        }

        Matrix hiddenLayerOutputs = new Matrix(network.getNumHiddenNodes(), 1);

        for (Digit digit : digits) {
            network.evaluate(digit.getInputVector(), hiddenLayerOutputs);

            for (int i = 0; i < queues.size(); i++) {
                Score score = new Score(digit, hiddenLayerOutputs.get(i, 0));
                queues.get(i).add(score);
            }
        }

        for (int i = 0; i < queues.size(); i++) {
            Iterator<Score> iterator = queues.get(i).iterator();
            for (int j = 0; j < 3; j++) {
                Score s = iterator.next();
            }

            BufferedImage masterImage = new BufferedImage(28 * 30, 28 * 30, BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D graphics = masterImage.createGraphics();
            AffineTransformOp op = new AffineTransformOp(new AffineTransform(), AffineTransformOp.TYPE_BICUBIC);

            iterator = queues.get(i).iterator();
            for (int x = 0; x < 30; x++) {
                for (int y = 0; y < 30; y++) {
                    Score s = iterator.next();
                    AWTImage awtImage = ((AWTImage)s.digit.getImage());
                    BufferedImage bufferedImage = awtImage.getBufferedImage();
                    graphics.drawImage(bufferedImage, op, 28 * x, 28 * y);
                }
            }

            graphics.dispose();

            ImageIO.write(masterImage, "png", new File(args[2] + i + ".png"));
        }
    }
}
