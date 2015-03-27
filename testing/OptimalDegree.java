package testing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.*;

import weka.core.matrix.Matrix;
import classification.*;

public class OptimalDegree {

	/**
	 * We have to find the degree for the polynomial kernel perceptron at which
	 * we observe best accuracy on the given data set 
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// Max Epochs ~ best from the OptimalEpochs.java
		int maxEpochs = 10; //5;
		// given range for the degree
		int[] degrees = {2, 3, 4, 5 ,6};

		// Test for accuracy for each degree
		for(int d=0; d<degrees.length; d++) {	// let d for degrees
			// initialize the thread service
			final ExecutorService service = Executors.newFixedThreadPool(10);
			// List to store the accuracies for each digit
			List<Future<int[]>> accuList = new ArrayList<Future<int[]>>();
			// total aggregate accuracy counts
			int[] aggregateCounts = {0, 0};
			
			// Now learn for each digit (on training data) & test on same digit (on development data)
			for(int n=0; n<10; n++) {	// n for number
				accuList.add(service.submit(new DigitDegreeThread(maxEpochs, degrees[d], n)));
			}
			
			// Aggregate the accuracies
			try{
				for(int n=0; n<10; n++) {
					aggregateCounts[0] += accuList.get(n).get()[0];
					aggregateCounts[1] += accuList.get(n).get()[1];
				}
			} catch(final InterruptedException ex) {
				ex.printStackTrace();
			} catch(final ExecutionException ex) {
				ex.printStackTrace();
			}
			
			// shutdown service
			service.shutdown();
			
			// print the accuracies
			System.out.println("Success Count: "+aggregateCounts[0]+" Total Counts: "+aggregateCounts[1]);
			System.out.println("Accuracy: "+ aggregateCounts[0]*100/(double) aggregateCounts[1]);
		}
	}

}

class DigitDegreeThread implements Callable<int[]> {
	
	// digit
	private int digit;
	// degree for the polynomial
	private int degree;
	// maxEpochs at which we have to train
	private int maxEpochs;
	
	// constructor
	public DigitDegreeThread(int maxEpochs, int degree, int digit) {
		this.digit= digit;
		this.degree = degree;
		this.maxEpochs = maxEpochs;
	}

	// implement call method
	public int[] call() throws IOException {
		
		// Get the kernel perceptron object
		KernelPerceptron kp = new KernelPerceptron();
		
		// Get the training matrices 
		GenerateMatrices gmTrain = new GenerateMatrices("./data/Digit"+digit+".tra");
		kp.featureVectors = gmTrain.getPHI();
		kp.mLabels = gmTrain.getLabelsVector();
		kp.normalize = true;
		kp.orderOfPolynomial = degree;
		
		// train the perceptron
		Matrix mAlpha = kp.trainKernelPerceptron(maxEpochs, Kernels.POLYNOMIAL, degree);
		
		// Get the matrices from development data for testing
		GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit"+digit+".dev");
		
		//--
		Matrix testPHI = gmDevelopment.getPHI();
		Matrix trueLabels = gmDevelopment.getLabelsVector();
		Matrix machineLabels = new Matrix(1, trueLabels.getColumnDimension());
		
		// array to count total examples & success predictions
		int[] counts = {0, 0};
		
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			// Test only those points which are our digit i.e either 0/1/2/..
			if(trueLabels.get(0, e)==1) {
				// Get the point for testing
				Matrix vNewPointFeatures = testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e);
				// test it for classification
				double result = kp.discriminantFunction(mAlpha, kp.mLabels, kp.featureVectors, vNewPointFeatures, Kernels.POLYNOMIAL);
				if(result>=0) {
					machineLabels.set(0, e, 1);
					counts[0]++;
				}
				else {
					machineLabels.set(0, e, -1);
				}
				// increase the total digit count
				counts[1]++;
			}
			else
				machineLabels.set(0, e, -1);
		}
		//--
		
//		Matrix machineLabels = kp.classify(mAlpha, gmTrain.getLabelsVector(), gmTrain.getPHI(),
//				gmDevelopment.getPHI(), Kernels.POLYNOMIAL, degree);
		
//		// array to count total examples & success predictions
//		int[] counts = {0, 0};
//		
//		// Count no of successful predictions & total examples
//		for(int e=0; e<machineLabels.getColumnDimension(); e++) {
//			// Count no of examples
//			if(gmDevelopment.getLabelsVector().get(0, e) == 1) {
//				counts[1]++;
//				// Count no of successful predictions
//				if(machineLabels.get(0, e) == 1) {
//					counts[0]++;
//				}
//			}
//		}
		// print counts
		System.out.println("Digit "+ digit +" Counts: "+ counts[0]+ "/" + counts[1]);
		// return the counts
		return counts;
	}
}
