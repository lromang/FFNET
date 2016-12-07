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

    /* Public variables
     * public static int   nHLayers   = 4;
     * public static int[] widthLayer = {3, 5, 7, 4, 1}; // Includes input and output layer
    */

    /*
     * ---------------------------------------------
     * Generate initial weights
     * ---------------------------------------------
     * x = observation
     */
    private static Matrix[] initW(int nHLayers, int[] widthLayer){
        Matrix[] layerWeights = new Matrix[nHLayers + 1];
        for(int i = 0; i < (nHLayers + 1); i++){
            layerWeights[i] = Matrix.random(widthLayer[i + 1], widthLayer[i] + 1);
        }
        return layerWeights;
    }

    /*
     * ---------------------------------------------
     * Generate initial observation (Should be
     * provided by Turing's Machine history)
     * ---------------------------------------------
     * x = observation
     */
    private static double[] initX(int inputLayerDim){
        double[] x = new double[inputLayerDim];
        for(int i = 0; i < inputLayerDim; i++){
            x[i]  = Math.random();
        }
        return x;
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
        double pred = W.times(b).get(0, 0);
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
     * Run the neural net activation option
     * ---------------------------------------------
     * layers = array of layer's weights
     * x      = observation
     */
    private static double runNet(Matrix[] layers, double[] x, char activation){
        // Layers[0] must be the weight matrix for the input vector.
        double[] hiddenOutput = hiddLayer(layers[0].getArray(), x, activation);
        // layers.length - 2 must be the number of hidden layers.
        for(int i = 1; i < (layers.length - 1); i++){
            hiddenOutput = hiddLayer(layers[i].getArray(), hiddenOutput, activation);
        }
        return outputLayer(layers[layers.length - 1].getArray(), hiddenOutput, activation);
    }

    /*
     * ---------------------------------------------
     * Run the neural net default logistic
     * ---------------------------------------------
     * layers = array of layer's weights
     * x      = observation
     */
    private static double runNet(Matrix[] layers, double[] x){
        double[] hiddenOutput = hiddLayer(layers[0].getArray(), x, 'l');
        for(int i = 1; i < (layers.length - 1); i++){
            hiddenOutput = hiddLayer(layers[i].getArray(), hiddenOutput, 'l');
        }
        return outputLayer(layers[layers.length - 1].getArray(), hiddenOutput, 'l');
    }

    /*
     * ---------------------------------------------
     * MAIN
     * ---------------------------------------------
     */
    public static void main(String args[]){
        Scanner scanner  = new Scanner(System.in);
        System.out.println("===========================================");
        System.out.println("========== Oracle Turing Machine ==========");
        System.out.println("===========================================\n");
        System.out.println("\nPlease enter the number of Hidden Layers of the Net: ");
        int nHLayers     = Integer.parseInt(scanner.next()); // This are only the hidden layers
        int[] widthLayer = new int[nHLayers + 2]; // This includes input
        // Fill in each layer.
        // INPUT LAYER
        System.out.println("\nPlease enter dimension of the input  layer: ");
        widthLayer[0] =  Integer.parseInt(scanner.next());
        // HIDDEN LAYERS
        for(int i = 1; i < (nHLayers + 1); i++){
            System.out.println("\nPlease enter dimension of the hidden layer " + i + " :");
            widthLayer[i] =  Integer.parseInt(scanner.next());
        }
        // Output Layer always 1
        widthLayer[nHLayers + 1] = 1;
        // Generate weights
        Matrix[] layers = initW(nHLayers, widthLayer);
        double[] x      = initX(widthLayer[0]);
        System.out.println("Network Output: " + runNet(layers, x, 't'));
    }
}
