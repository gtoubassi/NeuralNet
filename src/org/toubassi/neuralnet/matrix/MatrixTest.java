package org.toubassi.neuralnet.matrix;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 */
public class MatrixTest {

    @Test
    public void testMatrixTimesVector() {
        float data[][] = {{1, -1, 2}, {0, -3, 1}};
        Matrix matrix = new Matrix(data);
        float data2[][] = {{2}, {1}, {0}};
        Matrix vector = new Matrix(data2);
        Matrix m = matrix.times(vector);
        assertEquals(2, m.getRows());
        assertEquals(1, m.getCols());
        assertEquals(1f, m.get(0, 0), .00001);
        assertEquals(-3f, m.get(1, 0), .00001);
    }

    @Test
    public void testMatrixTimesMatrix() {
        float data1[][] = {{1, -1, 2}, {0, -3, 1}};
        Matrix matrix1 = new Matrix(data1);
        float data2[][] = {{2, 3}, {4, 5}, {0, 6}};
        Matrix matrix2 = new Matrix(data2);

        Matrix m = matrix1.times(matrix2);
        assertEquals(2, m.getRows());
        assertEquals(2, m.getCols());
        assertEquals(-2f, m.get(0, 0), .00001);
        assertEquals(10f, m.get(0, 1), .00001);
        assertEquals(-12f, m.get(1, 0), .00001);
        assertEquals(-9f, m.get(1, 1), .00001);
    }

    @Test
    public void testPlus() {
        float data1[][] = {{1, -1, 2}, {0, -3, 1}};
        Matrix matrix1 = new Matrix(data1);
        float data2[][] = {{4, 0, -4}, {4, 1, 5}};
        Matrix matrix2 = new Matrix(data2);

        Matrix sum = matrix1.plus(matrix2);
        assertEquals(2, sum.getRows());
        assertEquals(3, sum.getCols());
        assertEquals(5f, sum.get(0, 0), .00001);
        assertEquals(-1f, sum.get(0, 1), .00001);
        assertEquals(-2f, sum.get(0, 2), .00001);
        assertEquals(4f, sum.get(1, 0), .00001);
        assertEquals(-2f, sum.get(1, 1), .00001);
        assertEquals(6f, sum.get(1, 2), .00001);
    }

    @Test
    public void testMinus() {
        float data1[][] = {{1, -1, 2}, {0, -3, 1}};
        Matrix matrix1 = new Matrix(data1);
        float data2[][] = {{4, 0, -4}, {4, 1, 5}};
        Matrix matrix2 = new Matrix(data2);

        Matrix sum = matrix1.minus(matrix2);
        assertEquals(2, sum.getRows());
        assertEquals(3, sum.getCols());
        assertEquals(-3f, sum.get(0, 0), .00001);
        assertEquals(-1f, sum.get(0, 1), .00001);
        assertEquals(6f, sum.get(0, 2), .00001);
        assertEquals(-4f, sum.get(1, 0), .00001);
        assertEquals(-4f, sum.get(1, 1), .00001);
        assertEquals(-4f, sum.get(1, 2), .00001);
    }

    @Test
    public void testScalarTimes() {
        float data[][] = {{1, -2}, {0, 3}};
        Matrix m = new Matrix(data);
        Matrix product = m.scalarTimes(3.5f);

        assertEquals(2, product.getRows());
        assertEquals(2, product.getCols());
        assertEquals(3.5f, product.get(0, 0), .00001);
        assertEquals(-7f, product.get(0, 1), .00001);
        assertEquals(0f, product.get(1, 0), .00001);
        assertEquals(10.5f, product.get(1, 1), .00001);
    }

    @Test
    public void testComponentPow() {
        float data[][] = {{1, -2}, {0, 3}};
        Matrix m = new Matrix(data);
        Matrix product = m.arrayPow(2f);

        assertEquals(2, product.getRows());
        assertEquals(2, product.getCols());
        assertEquals(1f, product.get(0, 0), .00001);
        assertEquals(4f, product.get(0, 1), .00001);
        assertEquals(0f, product.get(1, 0), .00001);
        assertEquals(9f, product.get(1, 1), .00001);
    }

    @Test
    public void testComponentSum() {
        float data[][] = {{1, -2}, {0, 3}};
        Matrix m = new Matrix(data);

        assertEquals(2f, m.sum(), .00001);
    }

    @Test
    public void testSetAll() {
        float data[][] = {{1, -2}, {0, 3}};
        Matrix m = new Matrix(data);

        assertEquals(2f, m.sum(), .00001);

        m.setAll(0.0f);
        assertEquals(0f, m.sum(), .00001);
    }

    @Test
    public void testMatrixTranspose() {
        float data[][] = {{1, -2}, {0, 3}};
        float transposedData[][] = {{1, 0}, {-2, 3}};
        Matrix m = new Matrix(data);
        Matrix transposedM = new Matrix(transposedData);

        assertTrue(m.transpose().equals(transposedM));
        assertTrue(m.transpose().transpose().equals(m));
    }

    @Test
    public void testVectorTranspose() {
        float data[][] = {{1, -2, 3}};
        float transposedData[][] = {{1}, {-2}, {3}};;
        Matrix m = new Matrix(data);
        Matrix transposedM = new Matrix(transposedData);

        assertTrue(m.transpose().equals(transposedM));
        assertTrue(m.transpose().transpose().equals(m));
    }

}
