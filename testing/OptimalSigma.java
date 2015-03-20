package testing;

import java.io.IOException;
import weka.core.matrix.Matrix;
import classification.*;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.*;

public class OptimalSigma {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// Max Epochs ~ best from the OptimalEpochs.java
		int maxEpochs = 5;
		// given range for the degree
		double[] sigmas = {0.5, 2, 3, 5, 10};
		// corresponding accuracies for the degrees
		double[] accuracies = new double[sigmas.length];

		// Test for accuracy for each degree
		for(int d=0; d<sigmas.length; d++) {	// let d for degrees

			// Get the kernel perceptron object
			KernelPerceptron kp = new KernelPerceptron();
			// count of correct predictions
			double successCount = 0;
			// total no of examples
			double totalCount = 0;

			// Learn each digit on Training data & test on Development data
			for(int n=0; n<10; n++) {	// let n be a number

				// Get the training matrices 
				GenerateMatrices gmTrain = new GenerateMatrices("./data/Digit"+n+".tra");
				kp.featureVectors = gmTrain.getPHI();
				kp.mLabels = gmTrain.getLabelsVector();
				kp.normalize = true;

				// train the perceptron
				Matrix mAlpha = kp.trainKernelPerceptron(maxEpochs, Kernels.GAUSSIAN, sigmas[d]);

				// Get the matrices from development data for testing
				GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit"+n+".dev");
				Matrix machineLabels = kp.classify(mAlpha, gmTrain.getLabelsVector(), gmTrain.getPHI(),
						gmDevelopment.getPHI(), Kernels.GAUSSIAN, sigmas[d]);

				// Count no of successful predictions & total examples
				for(int e=0; e<machineLabels.getColumnDimension(); e++) {
					// Count no of examples
					if(gmTrain.getLabelsVector().get(0, e) == 1) {
						totalCount++;
						// Count no of successful predictions
						if(machineLabels.get(0, e) == 1) {
							successCount++;
						}
					}
				}
			}

			// Write the accuracy
			accuracies[d] = successCount*100/totalCount;
			System.out.println("SuccessCount: " + successCount + " TotalCount: " + totalCount);
			System.out.println("Total accuracies: "+accuracies[d]);
		}
	}
}

class DigitThread implements Callable<int[]> {
	// digit on which learning is done
	private int digit;
	// Max no of epochs to train on
	private int maxEpochs;
	// sigma value for the gaussian kernel
	private double sigma;
	
	// constructor
	public DigitThread(int maxEpochs, double sigma, int digit) {
		this.digit = digit;
		this.maxEpochs = maxEpochs;
		this.sigma = sigma;
	}
	
	// call method or thread functionality
	public int[] call() throws IOException {
		// count no of successful predictions & total no of examples
		int[] count = {0, 0};
		
		// Get the kernel perceptron object
		KernelPerceptron kp = new KernelPerceptron();
		
		// Get the training matrices 
		GenerateMatrices gmTrain = new GenerateMatrices("./data/Digit"+digit+".tra");
		kp.featureVectors = gmTrain.getPHI();
		kp.mLabels = gmTrain.getLabelsVector();
		kp.normalize = true;

		// train the perceptron
		Matrix mAlpha = kp.trainKernelPerceptron(maxEpochs, Kernels.GAUSSIAN, sigma);

		// Get the matrices from development data for testing
		GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit"+digit+".dev");
		Matrix machineLabels = kp.classify(mAlpha, gmTrain.getLabelsVector(), gmTrain.getPHI(),
				gmDevelopment.getPHI(), Kernels.GAUSSIAN, sigma);

		// Count no of successful predictions & total examples
		for(int e=0; e<machineLabels.getColumnDimension(); e++) {
			// Count no of examples
			if(gmTrain.getLabelsVector().get(0, e) == 1) {
				count[1]++;
				// Count no of successful predictions
				if(machineLabels.get(0, e) == 1) {
					count[0]++;
				}
			}
		}
		
		// return the count
		return count;
	}
	
}
