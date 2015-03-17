package classification;

import weka.core.matrix.Matrix;

public class KernelPerceptron {

	// sigma value for the gaussian kernel
	public static double sigmaForGaussian=0;
	// Order of polynomial
	public static int orderOfPolynomial=0;

	/**
	 * Simple Two class Kernel perceptron training.
	 * Returns Alpha matrix if converged else returns single cell matrix with value -1
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param vlabels - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @param kType - kernel type as in Kernels.java enum
	 * @return Alpha matrix 
	 */
	public static Matrix trainKernelPerceptron(Matrix featureVectors, Matrix vlabels, int maxEpochs, Kernels kType) {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha
		Matrix mAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=0; i<noOfExamples; i++) {
				// discriminant function value
				double functionValue = discriminantFunction(mAlpha, vlabels, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i, i), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != vlabels.get(0, i)) {
					// labels are different. So, improve mAlpha
					mAlpha.set(i, 0, mAlpha.get(i, 0)+1);
					// mark converged as false
					converged = false;
				}

				// Debug prints
				//				System.out.println("Iteration: " +e+ "-" +i);
				//				System.out.println("function value: " + functionValue + " ti: " + labelVector.get(0, i));
				//				System.out.println("new mAlpha:");
				//				mAlpha.print(2, 1);
			}
		}

		// Check if Converged!
		if(converged) {
			System.out.println("Converged ~ in Kernel Perceptron (" + kType.toString() + ") @epoch: "+ (e-1) + " With Alpha:");
			mAlpha.print(2, 1);

			// Return the trained alpha matrix
			return mAlpha;
		}
		else { 
			System.out.println("Not Converged! ~ in Kernel Perceptron (" + kType.toString() + ")");

			// returns single cell matrix with value -1 as it is not converged
			return new Matrix(1,1,-1);
		}
	}

	/**
	 * Simple Two class Averaged Kernel perceptron training.
	 * Returns Alpha matrix if converged else returns single cell matrix with value -1
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param vlabels - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @param kType - kernel type as in Kernels.java enum
	 * @return avg Alpha matrix 
	 */
	public static Matrix trainAveragedKernelPerceptron(Matrix featureVectors, Matrix vlabels, int maxEpochs, Kernels kType) {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha
		Matrix mAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;
		// Average Alpha initialization
		Matrix mAvgAlpha = new Matrix(noOfExamples, 1, 0);
		// Total iteration count
		double iterCount=1;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=0; i<noOfExamples; i++) {
				// discriminant function value
				double functionValue = discriminantFunction(mAlpha, vlabels, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i, i), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != vlabels.get(0, i)) {
					// labels are different. So, improve mAlpha
					mAlpha.set(i, 0, mAlpha.get(i, 0)+1);
					// mark converged as false
					converged = false;
				}

				// total sum of alpha
				mAvgAlpha = mAvgAlpha.plus(mAlpha);
				iterCount++;
				
				// Debug prints
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("Iteration total count: "+iterCount);
//				System.out.println("function value: " + functionValue + " ti: " + labelVector.get(0, i));
//				System.out.println("new mAlpha:");
//				mAlpha.print(2, 1);
//				System.out.println("New Avg of Alphas:");
//				mAvgAlpha.print(2, 1);
			}
		}
		
		// Average the total sum of Alpha
		mAvgAlpha = mAvgAlpha.times(1/iterCount);

		// Check if Converged!
		if(converged) {
			System.out.println("Converged ~ in Kernel Perceptron (" + kType.toString() + ") @epoch: "+ (e-1) + " With Alpha:");
			mAvgAlpha.print(2, 1);

			// Return the trained alpha matrix
			return mAvgAlpha;
		}
		else { 
			System.out.println("Not Converged! ~ in Kernel Perceptron (" + kType.toString() + ")");

			// returns single cell matrix with value -1 as it is not converged
			return new Matrix(1,1,-1);
		}
	}

	/**
	 * Classifies the given test data set from given parameters
	 * @param mAlpha - alpha matrix, compute from the trainKernelPerceptron method
	 * @param trainingLabels - training data class labels
	 * @param trainingPHI - training data feature vectors matrix of all data
	 * @param testPHI - test data feature vectors matrix of all data
	 * @param kType - kernel type to be used. Should be used same in training & testing
	 * @return
	 */
	public static Matrix classify(Matrix mAlpha, Matrix trainingLabels, Matrix trainingPHI, Matrix testPHI, Kernels kType) {

		// Check if the sent data is related or not
		if(mAlpha.getColumnDimension() != trainingLabels.getRowDimension() || 
				testPHI.getRowDimension() != trainingPHI.getRowDimension()) {
			System.out.println("Check the input matrices carefully for classify method!");
			System.out.println("Alpha dimensions may not be compatible with Traning Label dimensions!");
			System.out.println("No of features for training data may not be equal to testing data features!");
			System.exit(-1);
		}

		// No of Test Points
		int noOfTestPoints = testPHI.getColumnDimension();

		// Create a labels matrix for test data prediction
		Matrix machineLabels = new Matrix(1, noOfTestPoints, 0);

		// Predict the sign for each test point
		for(int i=0; i<noOfTestPoints; i++) {
			machineLabels.set(0, i, sign(discriminantFunction(mAlpha, trainingLabels, trainingPHI,
					testPHI.getMatrix(0, testPHI.getRowDimension()-1, i, i), kType)));
		}

		// return the machine Labels
		return machineLabels;
	}

	/**
	 * Discriminant function value when using Kernel Perceptron
	 * @param mAlpha - vector of alpha values for each training example i.e Alpha_i
	 * @param vLabels - vector of all the class labels for the training examples
	 * @param phi - Matrix of feature vectors of all the training examples
	 * @param vNewPointFeatures - data point for which the prediction is needed
	 * @param kType - Kernel Type (as in Kernels.java enum)
	 * @return value of the prediction
	 */
	public static double discriminantFunction(Matrix mAlpha, Matrix vLabels, Matrix phi, Matrix vNewPointFeatures, Kernels kType) {
		// value
		double result=0;

		// Sum of combination of each point with new data point (alpha * label * KernelValue of the both points)
		for(int j=0; j<phi.getColumnDimension(); j++) {

			// Calculate the value only if alpha value is not zero
			if(mAlpha.get(j, 0) != 0) {
				// Kernel value
				double kernelValue=0;
				if(kType == Kernels.LINEAR) {
					kernelValue = linearKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewPointFeatures);
				}
				else if(kType == Kernels.QUADRATIC) {
					kernelValue = quadraticKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewPointFeatures);
				}
				else if(kType == Kernels.POLYNOMIAL) {
					kernelValue = polynomialKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewPointFeatures);
				}
				else if(kType == Kernels.GAUSSIAN) {
					kernelValue = gaussianKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewPointFeatures);
				}
				else {
					System.out.println("Kernel type Unsupported!!");
					System.exit(-1);
				}
				// sum of(alpha_j * label_j * kernel(j, i)); where j = 1 .. N
				result += mAlpha.get(j, 0) * vLabels.get(0, j) * kernelValue;
			}
		}

		// return
		return result;
	}

	/**
	 * Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double linearKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		return vPhiJ.transpose().times(vNewPointFeatures).get(0, 0);
	}

	/**
	 * Quadratic Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double quadraticKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		return Math.pow(1+linearKernel(vPhiJ, vNewPointFeatures), 2);
	}

	/**
	 * Polynomial Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double polynomialKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		return Math.pow(1+linearKernel(vPhiJ, vNewPointFeatures), orderOfPolynomial);
	}

	/**
	 * Gaussian Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double gaussianKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		Matrix resultMatrix = vPhiJ.minus(vNewPointFeatures);
		return Math.exp(resultMatrix.transpose().times(resultMatrix).get(0, 0)/-2*sigmaForGaussian*sigmaForGaussian);
	}

	/**
	 * Find the sign of the value. i.e either +ve or -ve
	 * @param value
	 * @return 1 - if 0 or +ve / -1 if -ve
	 */
	private static int sign(double value) {
		if(value >= 0)
			return 1;
		else
			return -1;
	}
}
