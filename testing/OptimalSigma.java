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
		double[] sigmas = {0.5}; //, 2, 3, 5, 10};
		// corresponding accuracies for the degrees
		double[] accuracies = new double[sigmas.length];

		// Test for accuracy for each degree
		for(int d=0; d<sigmas.length; d++) {	// let d for degrees
			// Initialize thread execution service
			final ExecutorService service = Executors.newFixedThreadPool(10);
			// list to store the accuracies from each digit
			List<Future<int[]>> accuList = new ArrayList<Future<int[]>>();
			
			// count no of successful predictions & total no of examples
			int[] aggrigateCount = {0, 0};
			
			// Send each digit learning & testing onto each thread
			for(int n=0; n<10; n++) {	// let n be a number
				accuList.add(service.submit(new DigitThread(maxEpochs, sigmas[d], n)));
			}
			
			// add the aggregate format
			try{
				for(int n=0; n<10; n++) {
					int[] successCounts = accuList.get(n).get();	// Get the values
					aggrigateCount[0] += successCounts[0];	// Add the success predictions
					aggrigateCount[1] += successCounts[1];	// add total count of examples
				}
			} catch(final InterruptedException ex) {
				ex.printStackTrace();
			} catch(final ExecutionException ex) {
				ex.printStackTrace();
			}
			
			// shutdown the service
			service.shutdownNow();
			
			// Write the accuracy
			accuracies[d] = aggrigateCount[0]*100/(double)aggrigateCount[1];
			System.out.println("SuccessCount: " + aggrigateCount[0] + " TotalCount: " + aggrigateCount[1]);
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
		// Get the kernel perceptron object
		KernelPerceptron kp = new KernelPerceptron();
		kp.sigmaForGaussian = sigma;
		
		// Get the training matrices 
		GenerateMatrices gmTrain = new GenerateMatrices("./data/Digit"+digit+".tra");
		kp.featureVectors = gmTrain.getPHI();
		kp.mLabels = gmTrain.getLabelsVector();
		kp.normalize = true;

		// train the perceptron
		Matrix mAlpha = kp.trainKernelPerceptron(maxEpochs, Kernels.GAUSSIAN, sigma);

		// Get the matrices from development data for testing
		GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit"+digit+".dev");
		
		//--
		// assign the development matrices
		Matrix testPHI = gmDevelopment.getPHI();
		Matrix trueLabels = gmDevelopment.getLabelsVector();
		Matrix machineLabels = new Matrix(1, trueLabels.getColumnDimension());

		// count no of successful predictions & total no of examples
		int[] count = {0, 0};

		// test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			
			// For testing why disc value is 0
//			if((e==9 || e==18 || e==21 || e==29 || e==49) && digit==8){
//				System.out.println("@ zero giving example");
//			}
			
			// Test only those points which are our interested digit i.e either 0/1/2/..
			if(trueLabels.get(0, e)==1) {
				// Get the point for testing
				Matrix vNewPointFeatures = testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e);
				// test it for classification
				double result = kp.discriminantFunction(mAlpha, kp.mLabels, kp.featureVectors, vNewPointFeatures, Kernels.GAUSSIAN);
				
				if(result>0) {
					machineLabels.set(0, e, 1);
					count[0]++;
				}
				else {
					machineLabels.set(0, e, -1);
				}
				// increase the total digit count
				count[1]++;
			}
			else
				machineLabels.set(0, e, -1);
		}
		//--
		
//		Matrix machineLabels = kp.classify(mAlpha, gmTrain.getLabelsVector(), gmTrain.getPHI(),
//				gmDevelopment.getPHI(), Kernels.GAUSSIAN, sigma);
//
//		// Count no of successful predictions & total examples
//		for(int e=0; e<machineLabels.getColumnDimension(); e++) {
//			// Count no of examples
//			if(gmDevelopment.getLabelsVector().get(0, e) == 1) {
//				count[1]++;
//				// Count no of successful predictions
//				if(machineLabels.get(0, e) == 1) {
//					count[0]++;
//				}
//			}
//		}
		
		// print counts
		System.out.println("Digit "+ digit +" Counts: "+ count[0]+ "/" + count[1]);
		
		// return the count
		return count;
	}
	
}
