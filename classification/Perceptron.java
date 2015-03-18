package classification;

import weka.core.matrix.Matrix;

public class Perceptron {

	// Member Variables
	// PHI (vector of feature vectors for each input element in data-set)
	public Matrix featureVectors;
	// vector of labels for each input data
	public Matrix labelsVector;
	// is converged or not
	private boolean converged = false;
	
	/**
	 * Simple Two class perceptron training. 
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @return W matrix 
	 */
	public Matrix trainPerceptron(int maxEpochs) {

		// Create a weight vector matrix, W - Dim: rows in (featureVectors) X 1
		// i.e no. of features in each element X 1 & Initialize it to zero
		Matrix vW = new Matrix(featureVectors.getRowDimension(), 1, 0);
		
		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=0; i<noOfExamples; i++) {
				// phiXi = feature vector of i th element ; Matrix starts from 0 to n-1 row/column
				Matrix phiXi = featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i, i);

				// Sign of the wT * phi(Xi)
				int yi = sign(discriminantFunction(vW, phiXi));

				// Check if predicted label is same as true label
				if(yi != labelsVector.get(0, i)) {
					// improve vW i.e vW = vW + ti*phi(Xi)
					vW = vW.plus(phiXi.times(labelsVector.get(0, i)));
					// mark converged as false
					converged = false;
				}
				
				// Debug prints
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("phiXi:");
//				phiXi.print(2, 1);
//				System.out.println("yi: " + yi + " ti: " + labelVector.get(0, i));
//				System.out.println("new W:");
//				vW.print(2, 1);
			}
		}
		
		// Check if Converged!
		if(converged) {
			setConverged(true);
			System.out.println("Converged. ~ in Simple Perceptron");
			System.out.println("@epoch: "+(e-1)+" for W:");
			vW.print(5, 1);
		}
		else {
			System.out.println("Not Converged! ~ in Simple Perceptron");
		}
		
		// Return the W vector calculated for prediction
		return vW;
	}
	
	/**
	 * Simple Two class average perceptron training. 
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @return avg W matrix 
	 */
	public Matrix trainAveragedPerceptron(int maxEpochs) {

		// Create a weight vector matrix, W - Dim: rows in (featureVectors) X 1
		// i.e no. of features in each element X 1 & Initialize it to zero
		Matrix vW = new Matrix(featureVectors.getRowDimension(), 1, 0);
		
		// Create a average weight vector matrix, avgW - Dim: same as W
		Matrix vAvgW = new Matrix(vW.getRowDimension(), vW.getColumnDimension(), 0);
		
		// count for total iterations on the data
		// To get correct values in division, use count datatype as 'double'
		double iterCount = 1;

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// phiXi = feature vector of i th element ; Matrix starts from 0 to n-1 row/column
				Matrix phiXi = featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i-1, i-1);

				// Sign of the wT * phi(Xi)
				int yi = sign(discriminantFunction(vW, phiXi));

				// Check if predicted label is same as true label
				if(yi != labelsVector.get(0, i-1)) {
					// improve vW i.e vW = vW + ti*phi(Xi)
					vW = vW.plus(phiXi.times(labelsVector.get(0, i-1)));
					// mark converged as false
					converged = false;
				}
				
				// Sum the weight vector in each iteration
				vAvgW = vAvgW.plus(vW);
				// increase the total iteration count
				iterCount++;
				
				// Debug prints
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("phiXi:");
//				phiXi.print(2, 1);
//				System.out.println("yi: " + yi + " ti: " + labelsVector.get(0, i-1));
//				System.out.println("new W:");
//				vW.print(2, 1);
//				System.out.println("new avg W:");
//				vAvgW.print(2, 1);
//				System.out.println("iteration Count: "+iterCount);
//				System.out.println("--------------");
			}
		}

		// Average the total sum of W's
		vAvgW = vAvgW.times(1/iterCount);
		
		// Check if Converged!
		if(converged) {
			setConverged(true);
			System.out.println("Converged. ~ in averaged perceptron. (Converged for w not for avgW)");
			System.out.println("@epoch: "+(e-1)+" for average vector W:");
			vAvgW.print(5, 5);
		}
		else {
			System.out.println("Not Converged! ~ in Averaged Perceptron");
		}
		
		// Return the average weight Vector
		return vAvgW;
	}
	
	/**
	 * Classifies the given test data into different classes either +1 or -1
	 * @param w	- learned weight vector from training data
	 * @param testPHI - matrix of all feature vectors from the test data set
	 * @return labels matrix i.e matrix with label for each test example in each column. Dim: 1 x noOfTestExamples 
	 */
	public Matrix classify(Matrix w, Matrix testPHI) {
		
		// Test if the given matrices are compatible or not
		if(w.getRowDimension() != testPHI.getRowDimension()) {
			System.out.println("Check the input matrices carefully for classify method!");
			System.out.println("No of Features in test examples is not compatible with 'W' matrix dimensions! They must be equal.");
			System.exit(-1);
		}
		
		// No of test data points
		int noOfTestPoints = testPHI.getColumnDimension();
		
		// Create a class label matrix to write the predictions
		Matrix vLabels = new Matrix(1, noOfTestPoints, 0);
		
		// Predict the class for each test point
		for(int i=0; i<noOfTestPoints; i++) {
			vLabels.set(0, i, sign(discriminantFunction(w, 
					testPHI.getMatrix(0, testPHI.getRowDimension()-1, i, i))));
		}
		
		// return labels
		return vLabels;
	}
	
	/**
	 * Discriminant function value when using simple Perceptron
	 * @param vW - W matrix (learned weight vector)
	 * @param phiXi - Feature vector of example i
	 * @return vW transpose * phiXi value
	 */
	public double discriminantFunction(Matrix vW, Matrix phiXi) {
		// return f(x) = wT * phi(Xi)
		return vW.transpose().times(phiXi).get(0, 0);
	}
	
	/**
	 * Find the sign of the value. i.e either +ve or -ve
	 * @param value
	 * @return 1 - if 0 or +ve / -1 if -ve
	 */
	private int sign(double value) {
		if(value >= 0)
			return 1;
		else
			return -1;
	}
	
	/**
	 * Set the converged variable to true or false
	 * @param flag
	 */
	private void setConverged(boolean flag) {
		converged = flag;
	}
	
	/**
	 * Get if the model converged or not on given data
	 * @return true if converged else false
	 */
	public boolean isConverged() {
		return converged;
	}
}
