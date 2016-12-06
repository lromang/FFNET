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
    private static double[] addBias(double[] x){
        int rowLength = x.length;
        double[] newX = new double[rowLength + 1];
        for(int i = 0; i < rowLength; i++){
            newX[i] = x[i];
        }
        newX[rowLength] = 1;
        return newX;
    }

    /*
     * ---------------------------------------------
     * Activation layer
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static Matrix actLayer(double[][] w, double[] x){
        Matrix W        = new Matrix(w);
        Matrix b        = new Matrix(addBias(x), 1).transpose();
        Matrix H        = W.times(b);
        // Number of hidden layers
        int n_hidden    = H.getRowDimension();
        double[] hidden = new double[n_hidden];
        // Apply tanh
        for(int i = 0; i < n_hidden; i++){
            hidden[i] = Math.tanh(H.get(i, 0));
        }
        return H;
    }

    public static void main(String args[]){
        double[][] A   = {{.1, .2, .3, .1}, {.4, .5, .6, .2}, {.7, .8, .9, .3}, {.5, .5, .8, .2}};
        double[] b     = {.1, .2, .3};
        double[] one   = {1, 1, 1};
        Matrix oneM    = new Matrix(one, 1);
        Matrix bM      = new Matrix(b, 1).transpose();
        Matrix AM      = new Matrix(A);
        // Matrix AB      = AM.times(bM);
        Matrix H       = actLayer(A, b);
        System.out.println("Dimension columnas b");
        System.out.println(H.getColumnDimension());
        System.out.println("Dimension renglones b");
        System.out.println(H.getRowDimension());

    }
}
