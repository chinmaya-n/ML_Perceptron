package testing;

import java.io.IOException;
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
		int maxEpochs = 5;
		// given range for the degree
		int[] degrees = {2, 3, 4, 5 ,6};
		// corresponding accuracies for the degrees
		double[] accuracies = new double[degrees.length];

		// Test for accuracy for each degree
		for(int d=0; d<degrees.length; d++) {	// let d for degrees

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
				Matrix mAlpha = kp.trainKernelPerceptron(maxEpochs, Kernels.POLYNOMIAL, degrees[d]);
				
				// Get the matrices from development data for testing
				GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit"+n+".dev");
				Matrix machineLabels = kp.classify(mAlpha, gmTrain.getLabelsVector(), gmTrain.getPHI(),
						gmDevelopment.getPHI(), Kernels.POLYNOMIAL, degrees[d]);
				
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
