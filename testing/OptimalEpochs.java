package testing;
import java.io.IOException;

import weka.core.matrix.Matrix;
import classification.*;

public class OptimalEpochs {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		int epochs[] = {1, 5, 10, 20}; 	// epoch array
		double[] accuracy = new double[epochs.length];	// accuracy %ge for each epoch in the epochs array 
 
		// For each epoch find the accuracy on development data
		for(int i=0; i<epochs.length; i++) {
			
			// Create Perceptron object to compute
			Perceptron perceptron = new Perceptron();
			// count of successful predictions in each epoch
			double successCount = 0;
			// total no of examples
			double totalCount = 0;
			
			// Train for every digit
			for(int j=0; j<10; j++) {
				
				// Get the Training Matrices from data file
				GenerateMatrices gmTraining = new GenerateMatrices("./data/Digit" + j + ".tra");
				// populate in perceptron
				perceptron.featureVectors = gmTraining.getPHI();
				perceptron.labelsVector = gmTraining.getLabelsVector();
				// train the perceptron to get W
				Matrix mW = perceptron.trainPerceptron(epochs[i]);
				
				// Normalize the weight vector
				mW = mW.times(1/(mW.transpose().times(mW).get(0,0)));
//				mW.print(1, 10);
				
				// Now test the learned weight vector on development data
				// Get development data
				GenerateMatrices gmDevelopment = new GenerateMatrices("./data/Digit" + j + ".dev");
				// Get the system labels for development data
				Matrix machineLabels = perceptron.classify(mW, gmDevelopment.getPHI());
				
				// Count no of successful predictions
				for(int n=0; n<machineLabels.getColumnDimension(); n++) {
					// check for the current class examples 
					if(gmDevelopment.getLabelsVector().get(0, n) == 1) {
						totalCount++;	// increase total count
						// check the machine label predicted for the same input data point
						if(machineLabels.get(0, n) == 1) {
							successCount++;	// increment success count 
						}
					}
				}
			}
			
			// Now write the accuracy for this epoch value
			accuracy[i] = (successCount * 100)/totalCount;
			System.out.println("Successful Predictions: " + successCount + " Total Count: " + totalCount);
			System.out.println("Accuracy for epoch value: " + epochs[i] + " is : " + accuracy[i]);
		}
	}

}
