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

    // Public variables
    public static int   nHLayers    = 4;
    public static int[] widthLayer = {3, 5, 7, 4, 1}; // Includes input layer

    /*
     * ---------------------------------------------
     * Generate initial weights
     * ---------------------------------------------
     * x = observation
     */
    private static Matrix[] initW(){
        Matrix[] layerWeights = new Matrix[nHLayers];
        for(int i = 0; i < nHLayers; i++){
            layerWeights[i] = Matrix.random(widthLayer[i + 1], widthLayer[i] + 1);
        }
        return layerWeights;
    }

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
     * Hidden layer with activation option
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static double[] hiddLayer(double[][] w, double[] x, char activation){
        Matrix W        = new Matrix(w);
        Matrix b        = new Matrix(addBias(x), 1).transpose();
        Matrix H        = W.times(b);
        // Number of hidden layers
        int n_hidden    = H.getRowDimension();
        double[] hidden = new double[n_hidden];
        // Apply tanh
        for(int i = 0; i < n_hidden; i++){
            if(activation == 'l'){
                hidden[i] = Math.log(H.get(i, 0));
            }else{
                hidden[i] = Math.tanh(H.get(i, 0));
            }
        }
        return hidden;
    }

    /*
     * ---------------------------------------------
     * Hidden layer with default logistic
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static double[] hiddLayer(double[][] w, double[] x){
        return hiddLayer(w, x, 'l');
    }


    /*
     * ---------------------------------------------
     * Output layer with activation option
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static double outputLayer(double[][] w, double[] x, char activation){
        Matrix W    = new Matrix(w);
        Matrix b    = new Matrix(addBias(x), 1).transpose();
        double pred = W.times(b).get(1, 1);
        if(activation == 'l'){
            return Math.log(pred);
        }else{
            return Math.tanh(pred);
        }
    }

    /*
     * ---------------------------------------------
     * Output layer with default logistic
     * ---------------------------------------------
     * w = weight matrix
     * x = observation
     */
    private static double outputLayer(double[][] w, double[] x){
        return outputLayer(w, x, 'l');
    }

    /*
     * ---------------------------------------------
     * MAIN
     * ---------------------------------------------
     */
    public static void main(String args[]){
        Matrix   W1 = Matrix.random(5, 4);
        Matrix   W2 = Matrix.random(5, 4);
        double[] x  = {.1, .1, .1};
        double[] y  = hiddLayer(W1.getArray(), x);
        Matrix[] layers = initW();
        for(int i = 0; i < layers.length; i++){
            System.out.println("Número de filas: " + layers[i].getRowDimension() + " | Número de columnas: " + layers[i].getColumnDimension());
        }
    }
}
