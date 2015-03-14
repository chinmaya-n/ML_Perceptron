package classification;

import weka.core.matrix.Matrix;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.StringTokenizer;

public class Perceptron {

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
		perceptron(vPHI, vT, 5);
		
		// send to average perceptron 
		averagedPerceptron(vPHI, vT, 5);
		
		// send to kernel perceptron
		kernelPerceptron(vPHI, vT, 9);
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
				int yi = sign(multiply(vW.transpose(), phiXi).get(0, 0));

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
		int iterCount = 1;

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
				int yi = sign(multiply(vW.transpose(), phiXi).get(0, 0));

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
//				System.out.println("Iteration: " +e+ "-" +i);
//				System.out.println("phiXi:");
//				phiXi.print(2, 1);
//				System.out.println("yi: " + yi + " ti: " + labelVector.get(0, i-1));
//				System.out.println("new W:");
//				vW.print(2, 1);
//				System.out.println("new avg W:");
//				vAvgW.print(2, 1);
//				System.out.println("iter Count: "+iterCount);
			}
		}

		// Check if Converged!
		if(converged) {
			System.out.println("Converged @ average vector W:");
			(vAvgW.times(1/iterCount)).print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		// Return the average weight Vector/iterCount
		return vAvgW.times(1/iterCount);
	}

	/**
	 * Simple Two class perceptron
	 * @param featureVectors - PHI (vector of feature vectors for each input element in data-set)
	 * @param labelVector - vector of labels for each input data
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @return Alpha matrix 
	 */
	public static Matrix kernelPerceptron(Matrix featureVectors, Matrix labelVector, int maxEpochs) {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha
		Matrix vAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;
		// Calculate the Kernel for given featureVectors
		Matrix linearKernel = kernel(featureVectors);
		linearKernel.print(3, 1);

		// Run for given Max no. of Epochs or until converged
		for(int e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=1; i<=noOfExamples; i++) {
				// function value i.e sum of(alpha_j * label_j * kernel(j, i)); where j = 1 .. N
				double functionValue = 0;
				for(int j=1; j<=noOfExamples; j++) {
					functionValue += vAlpha.get(j-1, 0) * labelVector.get(0, j-1) * linearKernel.get(j-1, i-1);
				}
				
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
				System.out.println("Iteration: " +e+ "-" +i);
				System.out.println("function value: " + functionValue + " ti: " + labelVector.get(0, i-1));
				System.out.println("new vAlpha:");
				vAlpha.print(2, 1);
			}
		}
		
		// Check if Converged!
		if(converged) {
			System.out.println("Converged:");
			vAlpha.print(2, 1);
		}
		else
			System.out.println("Not Converged!");

		return vAlpha;
	}
	
	/**
	 * Kernel value calculation for given phi matrix
	 * @param phi - matrix of all feature vectors - M (no of features) x N (no of examples)
	 * @return Kernel matrix - N x N dimension.
	 */
	private static Matrix kernel(Matrix phi) {
		return multiply(phi.transpose(), phi);
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
