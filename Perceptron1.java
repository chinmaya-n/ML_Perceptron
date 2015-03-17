import weka.core.matrix.Matrix;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.StringTokenizer;

public class Perceptron1 {

	public static void main(String[] argv) throws IOException {
		// Get the data file
		String dataFile = argv[0];

		// Get the number of lines the data file -> no of examples ( used in PHI, T matrix dimension )
		int noOfExamples = 0;
		LineNumberReader lnr = new LineNumberReader(new FileReader(dataFile));
		while(lnr.readLine() != null) {
			noOfExamples++;
		}
		lnr.close();

		// Read the file
		BufferedReader br = new BufferedReader(new FileReader(dataFile));
		String line = br.readLine();

		// Count no of features in each of the feature vectors. -> ( used in PHI matrix dimension )
		StringTokenizer tokenizer = new StringTokenizer(line, ",");
		int noOfFeatures = tokenizer.countTokens()-1;

		// Create PHI matrix with dimensions: noOfFeatures X noOfExamples
		Matrix vPHI = new Matrix(noOfFeatures, noOfExamples);

		// Create T matrix which is class classifier matrix
		Matrix vT = new Matrix(1, noOfExamples);

		// Fill the Matrix with given elements
		int e=0;
		do {
			// tokenize the given string with delimiter ,
			tokenizer = new StringTokenizer(line, ",");

			// Fill the PHI matrix i.e. feature vectors matrix
			for(int f=0; f<noOfFeatures; f++) {
				vPHI.set(f, e, Double.parseDouble(tokenizer.nextElement().toString()));
			}
			// Fill the T matrix i.e. class classifier
			vT.set(0, e, Double.parseDouble(tokenizer.nextElement().toString()));

			e++;
			line = br.readLine();
		} while(line != null && e<noOfExamples);

		// Close
		br.close();

		// send to perceptron
//		perceptron(vPHI, vT, 5);
		
		// send to average perceptron 
//		averagedPerceptron(vPHI, vT, 5);
		
		// send to kernel perceptron
		kernelPerceptron(vPHI, vT, 20, Kernels.POLYNOMIAL);
		
		// send to average kernel perceptron
//		averagedKernelPerceptron(vPHI, vT, 5, Kernels.POLYNOMIAL);
	}

	/**
	 * Simple Two class perceptron
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param labelVector - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @return W matrix 
	 */
	public static Matrix perceptron(Matrix featureVectors, Matrix labelVector, int maxEpochs) {

		// Create a weight vector matrix, W - Dim: rows in (featureVectors) X 1
		// i.e no. of features in each element X 1 & Initialize it to zero
		Matrix vW = new Matrix(featureVectors.getRowDimension(), 1, 0);

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		for(int e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// phiXi = feature vector of i th element ; Matrix starts from 0 to n-1 row/column
				Matrix phiXi = featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i-1, i-1);

				// Sign of the wT * phi(Xi)
				int yi = sign(predictPerceptron(vW, phiXi));

				// Check if predicted label is same as true label
				if(yi != labelVector.get(0, i-1)) {
					// improve vW
					vW = vW.plus(phiXi.times(labelVector.get(0, i-1)));
					// mark converged as false
					converged = false;
				}
				
				// Debug prints
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("phiXi:");
//				phiXi.print(2, 1);
//				System.out.println("yi: " + yi + " ti: " + labelVector.get(0, i-1));
//				System.out.println("new W:");
//				vW.print(2, 1);
			}
		}
		
		// Check if Converged!
		if(converged) {
			System.out.println("Converged:");
			vW.print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		return vW;
	}

	/**
	 * 
	 * @param featureVectors
	 * @param labelVector
	 * @param maxEpochs
	 * @return
	 */
	public static Matrix averagedPerceptron(Matrix featureVectors, Matrix labelVector, int maxEpochs) {

		// Create a weight vector matrix, W - Dim: rows in (featureVectors) X 1
		// i.e no. of features in each element X 1 & Initialize it to zero
		Matrix vW = new Matrix(featureVectors.getRowDimension(), 1, 0);
		
		// Create a avg weight vector matrix, avgW - Dim: same as W
		Matrix vAvgW = new Matrix(vW.getRowDimension(), vW.getColumnDimension(), 0);
		
		// count for total iterations on the data
		double iterCount = 1;

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		for(int e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// phiXi = feature vector of i th element ; Matrix starts from 0 to n-1 row/column
				Matrix phiXi = featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i-1, i-1);

				// Sign of the wT * phi(Xi)
				int yi = sign(predictPerceptron(vW, phiXi));

				// Check if predicted label is same as true label
				if(yi != labelVector.get(0, i-1)) {
					// improve vW
					vW = vW.plus(phiXi.times(labelVector.get(0, i-1)));
					// mark converged as false
					converged = false;
				}
				
				// Calculate the average weight vector
				vAvgW = vAvgW.plus(vW);
				// increase the total iteration count
				iterCount++;
				
				// Debug prints
				System.out.println("Iteration: " +e+ "-" +i);
				System.out.println("phiXi:");
				phiXi.print(2, 1);
				System.out.println("yi: " + yi + " ti: " + labelVector.get(0, i-1));
				System.out.println("new W:");
				vW.print(2, 1);
				System.out.println("new avg W:");
				vAvgW.print(2, 1);
				System.out.println("iter Count: "+iterCount);
			}
		}

		// Average the total sum of W's
		vAvgW = vAvgW.times(1/iterCount);
		
		// Check if Converged!
		if(converged) {
			System.out.println("Converged @ average vector W:");
			vAvgW.print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		// Return the average weight Vector/iterCount
		return vAvgW.times(1/iterCount);
	}
	
	/**
	 * Discriminant function value when using simple Perceptron
	 * @param vW - W matrix
	 * @param phiXi - Feature vector of example i
	 * @return vW transpose * phiXi value
	 */
	public static double predictPerceptron(Matrix vW, Matrix phiXi) {
		return multiply(vW.transpose(), phiXi).get(0, 0);
	}

	/**
	 * Simple Two class Kernel perceptron
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param labelVector - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @param kType - kernel type as in Kernels.java enum
	 * @return Alpha matrix 
	 */
	public static Matrix kernelPerceptron(Matrix featureVectors, Matrix labelVector, int maxEpochs, Kernels kType) {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha
		Matrix vAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// function value
				double functionValue = predictKernelPerceptron(vAlpha, labelVector, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i-1, i-1), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != labelVector.get(0, i-1)) {
					// labels are different. So, improve vAlpha
					vAlpha.set(i-1, 0, vAlpha.get(i-1, 0)+1);
					// mark converged as false
					converged = false;
				}
				
				// Debug prints
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("function value: " + functionValue + " ti: " + labelVector.get(0, i-1));
//				System.out.println("new vAlpha:");
//				vAlpha.print(2, 1);
			}
		}
		
		// Check if Converged!
		if(converged) {
			System.out.println("Converged @epoch: "+e);
			vAlpha.print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		return vAlpha;
	}
	
	/**
	 * Simple Two class averaged Kernel perceptron
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param labelVector - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @return Alpha matrix 
	 */
	public static Matrix averagedKernelPerceptron(Matrix featureVectors, Matrix labelVector, int maxEpochs, Kernels kType) {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha using no of examples count
		Matrix mAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;
		// Average Alpha initialization
		Matrix mAvgAlpha = new Matrix(noOfExamples, 1, 0);
		// Total iteration count
		double iterCount=1;
		
		// Run for given Max no. of Epochs or until converged
		for(int e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// function value
				double functionValue = predictKernelPerceptron(mAlpha, labelVector, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i-1, i-1), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != labelVector.get(0, i-1)) {
					// labels are different. So, improve vAlpha
					mAlpha.set(i-1, 0, mAlpha.get(i-1, 0)+1);
					// mark converged as false
					converged = false;
				}
				
				// total sum of alpha
				mAvgAlpha = mAvgAlpha.plus(mAlpha);
				iterCount++;
				
				// Debug prints
				System.out.println("Iteration: " +e+ "-" +i);
				System.out.println("Iteration total count: "+iterCount);
				System.out.println("function value: " + functionValue + " ti: " + labelVector.get(0, i-1));
				System.out.println("new vAlpha:");
				mAlpha.print(2, 1);
				System.out.println("New Avg of Alphas:");
				mAvgAlpha.print(2, 1);
			}
		}
		
		// Average the total sum of Alpha
		mAvgAlpha = mAvgAlpha.times(1/iterCount);
		
		// Check if Converged!
		if(converged) {
			System.out.println("Converged:");
			mAvgAlpha.print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		return mAvgAlpha;
	}
	
	/**
	 * Discriminant function value when using Kernel Perceptron
	 * @param vAlpha - vector of alpha values for each training example i.e Alpha_i
	 * @param vLabels - vector of all the class labels for the training examples
	 * @param phi - Matrix of feature vectors of all the training examples
	 * @param vNewDataPoint - data point for which the prediction is needed
	 * @param type - Kernel Type (as in Kernels.java enum)
	 * @return value of the prediction
	 */
	public static double predictKernelPerceptron(Matrix vAlpha, Matrix vLabels, Matrix phi, Matrix vNewDataPoint, Kernels type) {
		// value
		double result=0;
		
		// Sum of combination of each point with new data point (alpha * label * KernelValue of the both points)
		for(int j=0; j<phi.getColumnDimension(); j++) {
			
			// Calculate the value only if alpha value is not zero
			if(vAlpha.get(j, 0) != 0) {
				// Kernel value
				double kernelValue=0;
				if(type == Kernels.LINEAR) {
					kernelValue = linearKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewDataPoint);
				}
				else if(type == Kernels.QUADRATIC) {
					kernelValue = quadraticKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewDataPoint);
				}
				else if(type == Kernels.POLYNOMIAL) {
					kernelValue = polynomialKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewDataPoint);
				}
				else if(type == Kernels.GAUSSIAN) {
					kernelValue = gaussianKernel(phi.getMatrix(0, phi.getRowDimension()-1, j, j), vNewDataPoint);
				}
				else {
					System.out.println("Kernel type Unsupported!!");
					System.exit(-1);
				}
				// sum of(alpha_j * label_j * kernel(j, i)); where j = 1 .. N
				result += vAlpha.get(j, 0) * vLabels.get(0, j) * kernelValue;
			}
		}
		
		// return
		return result;
	}
	
	/**
	 * Kernel value calculation for given phi matrix
	 * @param phiJ - feature vector for example j
	 * @param phiI - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double linearKernel(Matrix vPhiJ, Matrix vPhiI) {
		return multiply(vPhiJ.transpose(), vPhiI).get(0, 0);
	}
	
	/**
	 * Quadratic Kernel value calculation for given phi matrix
	 * @param phiJ - feature vector for example j
	 * @param phiI - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double quadraticKernel(Matrix vPhiJ, Matrix vPhiI) {
		return Math.pow(1+multiply(vPhiJ.transpose(), vPhiI).get(0, 0), 2);
	}
	
	/**
	 * Polynomial Kernel value calculation for given phi matrix
	 * @param phiJ - feature vector for example j
	 * @param phiI - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double polynomialKernel(Matrix vPhiJ, Matrix vPhiI) {
		return Math.pow(1+multiply(vPhiJ.transpose(), vPhiI).get(0, 0), 4);
	}
	
	/**
	 * Polynomial Kernel value calculation for given phi matrix
	 * @param phiJ - feature vector for example j
	 * @param phiI - feature vector for data-point 
	 * @return Kernel value
	 */
	private static double gaussianKernel(Matrix vPhiJ, Matrix vPhiI) {
		// sigma
		double sigma=0;
		Matrix resultMatrix = vPhiJ.minus(vPhiI);
		return Math.exp(multiply(resultMatrix.transpose(), resultMatrix).get(0, 0)/-2*sigma*sigma);
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

	/**
	 * Multiply matrix A & B
	 * @param A
	 * @param B
	 * @return A*B matrix
	 */
	private static Matrix multiply(Matrix A, Matrix B) {
		// Three loops for each C[i][j] += A[i][k] * B[k][j]

		// Check dimensionality
		if(A.getColumnDimension() != B.getRowDimension()) {
			System.out.println("Error in dimention of the Matrix for multiplication!!");
			System.exit(-1);
		}

		// result Matrix
		Matrix C = new Matrix(A.getRowDimension(), B.getColumnDimension());

		for(int i=0; i<A.getRowDimension(); i++) {
			for(int j=0; j<B.getColumnDimension(); j++) {
				double cij = 0;
				for(int k=0; k<A.getColumnDimension(); k++) {
					// Find cij
					cij += A.get(i, k) * B.get(k, j);
				}

				// set cij
				C.set(i, j, cij);
			}
		}

		// return result
		return C;
	}
}
