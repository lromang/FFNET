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
     * Evaluate the neural net activation option on all
     * of the dataset.
     * ---------------------------------------------
     * layers = array of layer's weights
     * x      = observation
     */
    private static double[] runAllNet(double[] output, Matrix[] layers, double[][] x, char activation){
        double[] resNet = new double[x[0].length];
        for(int i = 0; i < resNet.length; i++){
            resNet[i] = (output[i] - runNet(layers, x[i], activation))*(output[i] - runNet(layers, x[i], activation));
        }
        return resNet;
    }


    /*
     * ---------------------------------------------
     * Run the neural net default logistic on all
     * of the dataset.
     * ---------------------------------------------
     * layers = array of layer's weights
     * x      = observation
     */
    private static double[] runAllNet(Matrix[] layers, double[][] x){
        return runAllNet(layers, x, 'l');
    }

    /*
     * ---------------------------------------------
     * Run the neural net default logistic (Only works
     * for two layered Neural Network).
     * ---------------------------------------------
     * layers = array of layer's weights
     * x      = observation
     * OBSERVATION: In order to make it batch gradient
     * descent, I need to sum over the delta over all
     * the observations or M observations for M batch
     * size.
     */
    private static Matrix[] runNetBackProp(Matrix[] layers, double[] x, int ToL, double outputValue, double learnRate){
        Matrix  W1 = layers[0];
        Matrix  W2 = layers[1];
        double[] y = hiddLayer(W1.getArray(),  x, 'l');
        double   z = outputLayer(W2.getArray(), y, 'l');
        double[] x_new = addBias(x);
        double[] y_new = addBias(y);
        for(int k = 0; k < ToL; k++){
            System.out.println("|pred - val|^2 = " + (z - outputValue)*(z - outputValue));
            /*-----------------------------*/
            // CALCULATE UPDATE W2
            double[] updateW2 = new double[W2.getColumnDimension()];
            for(int i = 0; i < W2.getColumnDimension(); i++){
                updateW2[i] = -learnRate*y_new[i]*(outputValue - z)*(1 - z*z);
            }
            // CALCULATE UPDATE W1
            int count = 1;
            double[][] updateW1 = new double[W1.getColumnDimension()][W1.getRowDimension()];
            for(int i = 0; i < W1.getColumnDimension(); i++){
                for(int j = 0; j < W1.getRowDimension(); j++){
                    updateW1[i][j] = -learnRate*(outputValue - z)*(1 - z * z)*W2.get(0, j)*(1 - y_new[j]*y_new[j])*x_new[i];
                    count++;
                }
            }
            /*-----------------------------*/
            // UPDATE
            Matrix MupdateW1 = new Matrix(updateW1).transpose();
            Matrix MupdateW2 = new Matrix(updateW2, 1);
            W1.plusEquals(MupdateW1);
            W2.plusEquals(MupdateW2);
            // Recalculate inner values.
            y = hiddLayer(W1.getArray(),  x, 'l');
            z = outputLayer(W2.getArray(), y, 'l');
            // Add bias
            y_new = addBias(y);
        }
        layers[0] = W1;
        layers[1] = W2;
        return layers;
    }


    /*
     * What we are missing:
     * - A function to run all the process of the neural net
     *     - Select a candidate observation.
     *     - Update weights.
     *     - Evaluate performance.
     * - Generate a Turing machine and a history of input
     *   output pairs:
     *     - Generate random Turing Machine
     *     - Feed multiple inputs and outputs
     * - Train Neural Network on contents of Turing machine
     * - Add Trained Neural Network to Turing Machine (Run Net at each iter)
     */



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
        // Get Tolerance
        System.out.println("\nPlease enter the maximum number of iterations: ");
        int ToL = Integer.parseInt(scanner.next());
        // Get learning rate
        System.out.println("\nPlease enter the learning rate: ");
        double learningRate = Double.parseDouble(scanner.next());
        // Run Neural Net
        runNetBackProp(layers, x, ToL, 1, learningRate);
    }
}
