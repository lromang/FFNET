/* #########################################
 * Luis Manuel Román García
 * luis.roangarci@gmail.com
 * #########################################
 *
 * -----------------------------------------
 * General pourpose routines for the
 * implementation of a feed forward neural
 * network.
 * -----------------------------------------
 *
 */

import java.util.*;
import java.util.Random;
import java.lang.Math;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
// Matrix manipulation
import Jama.*;


public class oracularTuring{

    /*
     * ---------------------------------------------
     * Add bias
     * ---------------------------------------------
     * x = observation
     */
    private static double[][] addBias(double[][] x){
        int rowLength = x[0].length;
        double[][] newX = new double[0][rowLength + 1];
        for(int i = 0; i < rowLength; i++){
            newX[0][i] = x[0][i];
        }
        newX[0][rowLength + 1] = 1;
        return newX;
    }

    /*
     * ---------------------------------------------
     * Activation layer
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static Matrix actLayer(double[][] w, double[][] x){
        Matrix W        = new Matrix(w);
        Matrix b        = new Matrix(x);
        Matrix H        = W.times(b);
        double[] hidden = new double[H.getRowDimension()];
        // Apply tanh
        for(int i = 0; i < H.getRowDimension(); i++){
            hidden[i] = Math.tanh(H.get(i, 0));
        }
        return H;
    }

    public static void main(String args[]){
        double[][] A   = {{.1, .2, .3, .4}, {.4, .5, .6, .7}, {.7, .8, .9, .9}};
        double[][] b   = {{.1, .2, .3}};
        double[][] one = {{1, 1, 1}};
        Matrix oneM    = new Matrix(one);
        Matrix bM      = new Matrix(b);
        Matrix AM      = new Matrix(A);
        //        Matrix H       = actLayer(A, b);
        System.out.println("Dimension columnas b");
        System.out.println(AM.getColumnDimension());
        System.out.println("Dimension renglones b");
        System.out.println(AM.getRowDimension());

    }
}
