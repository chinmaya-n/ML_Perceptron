package classification;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.StringTokenizer;

import weka.core.matrix.Matrix;

public class KernelPerceptron {

	// Member variables
	// PHI (vector of feature vectors for each input element in data-set)
	public Matrix featureVectors;
	// matrix of labels for each input data
	public Matrix mLabels;
	// Normalize the kernels or not
	public boolean normalize = false;

	// sigma value for the gaussian kernel
	public double sigmaForGaussian;
	// Order of polynomial
	public int orderOfPolynomial;
	// if converged or not
	private boolean converged = false;

	// write learned info to file
	public String writeLearnedInfoToFile = "";

	/**
	 * Simple Two class Kernel perceptron training.
	 * If using linear/quadratic kernels - send some dummy value for "kernelParam" parameter
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @param kType - kernel type as in Kernels.java enum
	 * @param kernelParam - sigma value / order of polynomial based on type of kernel
	 * @return Alpha matrix 
	 * @throws IOException 
	 */
	public Matrix trainKernelPerceptron(int maxEpochs, Kernels kType, double kernelParam) throws IOException {

		// Get the no of Examples from the data set
		int noOfExamples = featureVectors.getColumnDimension();
		// Create a matrix alpha
		Matrix mAlpha = new Matrix(noOfExamples, 1, 0);
		// set converged variable to false - just for beginning the loop
		boolean converged = false;

		// based on kType fill sigmaForGaussian / orderOfPolynomial
		if(kType == Kernels.POLYNOMIAL)
			orderOfPolynomial = (int) kernelParam;
		else if(kType == Kernels.GAUSSIAN)
			sigmaForGaussian = kernelParam;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=0; i<noOfExamples; i++) {
				// discriminant function value
				double functionValue = discriminantFunction(mAlpha, mLabels, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i, i), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != mLabels.get(0, i)) {
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
			setConverged(true);
			System.out.println("Converged ~ in Kernel Perceptron (" + kType.toString() + ") @epoch: "+ (e-1) + " With Alpha:");
//			mAlpha.print(2, 1);
		}
		else { 
			System.out.println("Not Converged! ~ in Kernel Perceptron (" + kType.toString() + ")");
		}

		// Check if learning write to file is enabled
		if(writeLearnedInfoToFile != "") {
			writeToFile(mAlpha, mLabels, featureVectors, writeLearnedInfoToFile);
		}

		// Return the trained alpha matrix
		return mAlpha;
	}

	/**
	 * Simple Two class Averaged Kernel perceptron training.
	 * If using linear/quadratic kernels - send some dummy value for "kernelParam" parameter
	 * @param maxEpochs - maximum number of epochs to perform if not converged 
	 * @param kType - kernel type as in Kernels.java enum
	 * @param kernelParam - sigma value / order of polynomial based on type of kernel
	 * @return avg Alpha matrix 
	 * @throws IOException 
	 */
	public Matrix trainAveragedKernelPerceptron(int maxEpochs, Kernels kType, double kernelParam) throws IOException {

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

		// based on kType fill sigmaForGaussian / orderOfPolynomial
		if(kType == Kernels.POLYNOMIAL)
			orderOfPolynomial = (int) kernelParam;
		else if(kType == Kernels.GAUSSIAN)
			sigmaForGaussian = kernelParam;

		// Run for given Max no. of Epochs or until converged
		int e;
		for(e=1; e<=maxEpochs && !converged; e++) {
			converged = true;

			// Run for each element in the data-set
			for(int i=0; i<noOfExamples; i++) {
				// discriminant function value
				double functionValue = discriminantFunction(mAlpha, mLabels, featureVectors, 
						featureVectors.getMatrix(0, featureVectors.getRowDimension()-1, i, i), kType);
				// Sign of the function value
				int yi = sign(functionValue);

				// Check if predicted label is same as true label
				if(yi != mLabels.get(0, i)) {
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
			setConverged(true);
			System.out.println("Converged ~ in Kernel Perceptron (" + kType.toString() + ") @epoch: "+ (e-1) + " With Alpha:");
//			mAvgAlpha.print(2, 1);
		}
		else { 
			System.out.println("Not Converged! ~ in Kernel Perceptron (" + kType.toString() + ")");
		}

		// Check if learning write to file is enabled
		if(writeLearnedInfoToFile != "") {
			writeToFile(mAvgAlpha, mLabels, featureVectors, writeLearnedInfoToFile);
		}

		// Return the trained alpha matrix
		return mAvgAlpha;
	}

	/**
	 * Classifies the given test data set from given parameters
	 * @param mAlpha - alpha matrix, compute from the trainKernelPerceptron method
	 * @param trainingLabels - training data class labels
	 * @param trainingPHI - training data feature vectors matrix of all data
	 * @param testPHI - test data feature vectors matrix of all data
	 * @param kType - kernel type to be used. Should be used same in training & testing
	 * @param kVariable - kernel variable i.e. sigma value / order of polynomial based on type of kernel
	 * @return machine labels matrix. Dim: 1 x noOfTestExamples
	 */
	public Matrix classify(Matrix mAlpha, Matrix trainingLabels, Matrix trainingPHI, Matrix testPHI, Kernels kType, double kVariable) {
		
		// Check if the sent data is related or not
		if(mAlpha.getColumnDimension() != trainingLabels.getRowDimension() || 
				testPHI.getRowDimension() != trainingPHI.getRowDimension()) {
			System.out.println("Check the input matrices carefully for classify method!");
			System.out.println("Alpha dimensions may not be compatible with Traning Label dimensions!");
			System.out.println("No of features for training data may not be equal to testing data features!");
			System.exit(-1);
		}
		
		// based on kType fill sigmaForGaussian / orderOfPolynomial
		if(kType == Kernels.POLYNOMIAL)
			orderOfPolynomial = (int) kVariable;
		else if(kType == Kernels.GAUSSIAN)
			sigmaForGaussian = kVariable;

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
	 * Classifies the given test data from the learned data file
	 * @param fileName - Learned data file (from the trainKernelPerceptron / trainKernelAveragedPerceptron)
	 * @param testPHI - test data feature vectors matrix of all data
	 * @param kType - kernel type to be used. Should be used same in training & testing
	 * @param kVariable - kernel variable i.e. sigma value / order of polynomial based on type of kernel
	 * @return machine labels matrix. Dim: 1 x noOfTestExamples
	 * @throws IOException
	 */
	public Matrix classify(String fileName, Matrix testPHI, Kernels kType, double kVariable) throws IOException {

		// Open the file to get the contents
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		
		// Get the size of the features for an example
		String line = br.readLine();
		StringTokenizer st = new StringTokenizer(line, " ");
		int noOfFeatures = st.countTokens()-1;	// Not counting Alpha*T (the first column)
		
		// Get the number of examples = no of lines in the given file
		int noOfExamples = 0;
		LineNumberReader lnr = new LineNumberReader(new FileReader(fileName));
		while(lnr.readLine() !=null) {
			noOfExamples++;
		}
		lnr.close();
		
		// Create matrices to store the contents from the file
		Matrix mAlphaT = new Matrix(noOfExamples, 1);
		Matrix mPHI = new Matrix(noOfFeatures, noOfExamples);
		Matrix mLabels = new Matrix(1, noOfExamples, 1);	// Dummy labels as the labels are already carried by mAlphaT
		
		// Read each line and populate the data
		int lineNumber = 0;
		do {
			// Tokenize the string
			StringTokenizer stLine = new StringTokenizer(line, " ");
			// Fill the Alpha*T value in mAlphaT matrix
			mAlphaT.set(lineNumber, 0, Double.parseDouble(stLine.nextToken()));
			// Fill the feature vectors in mPHI matrix
			for(int i=0; i<noOfFeatures; i++) {
				mPHI.set(i, lineNumber, Double.parseDouble(stLine.nextToken()));
			}
			
			// Read next line
			line = br.readLine();
			lineNumber++;
		}while(lineNumber < noOfExamples && line != null);
		
		// close buffer
		br.close();
		
		// Send the data to classify
		return classify(mAlphaT, mLabels, mPHI, testPHI, kType, kVariable);
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
	public double discriminantFunction(Matrix mAlpha, Matrix vLabels, Matrix phi, Matrix vNewPointFeatures, Kernels kType) {
		// value
		double result=0;

		// Sum of combination of each point with new data point (alpha * label * KernelValue of the both points)
		for(int j=0; j<phi.getColumnDimension(); j++) {

			// Calculate the value only if alpha value is not zero
			if(mAlpha.get(j, 0) != 0) {
				// Kernel value
				double kernelValue=0;
				Matrix trainingEg = phi.getMatrix(0, phi.getRowDimension()-1, j, j);
				if(kType == Kernels.LINEAR) {
					kernelValue = linearKernel(trainingEg, vNewPointFeatures);
				}
				else if(kType == Kernels.QUADRATIC) {
					orderOfPolynomial = 2;
					kernelValue = polynomialKernel(trainingEg, vNewPointFeatures);
				}
				else if(kType == Kernels.POLYNOMIAL) {
					kernelValue = polynomialKernel(trainingEg, vNewPointFeatures);
				}
				else if(kType == Kernels.GAUSSIAN) {
					kernelValue = gaussianKernel(trainingEg, vNewPointFeatures);
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
	private double linearKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		if(!normalize) {
			return vPhiJ.transpose().times(vNewPointFeatures).get(0, 0);
		}
		else {
			return vPhiJ.transpose().times(vNewPointFeatures).get(0, 0)/
					(Math.sqrt(vPhiJ.transpose().times(vPhiJ).get(0, 0) *
							vNewPointFeatures.transpose().times(vNewPointFeatures).get(0, 0)));
		}
	}

	/**
	 * Polynomial Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	private double polynomialKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		if(!normalize) {
			return Math.pow(1+vPhiJ.transpose().times(vNewPointFeatures).get(0, 0), orderOfPolynomial);
		}
		else {
			double numerator = Math.pow(1+vPhiJ.transpose().times(vNewPointFeatures).get(0, 0), orderOfPolynomial);
			double kernelXX = Math.pow(1+vPhiJ.transpose().times(vPhiJ).get(0, 0), orderOfPolynomial);
			double kernelYY = Math.pow(1+vNewPointFeatures.transpose().times(vNewPointFeatures).get(0, 0), orderOfPolynomial);
			double denominator = Math.sqrt( kernelXX * kernelYY);
//			System.out.println("kxx: "+kernelXX+ " kyy: "+kernelYY+ " Denominator: "+denominator+ " Numerator: " +numerator);
			return numerator/denominator;
		}
	}

	/**
	 * Gaussian Kernel value calculation for given phiJ matrix to new data point feature vector
	 * @param phiJ - feature vector for example j
	 * @param vNewPointFeatures - feature vector for data-point 
	 * @return Kernel value
	 */
	public double gaussianKernel(Matrix vPhiJ, Matrix vNewPointFeatures) {
		Matrix resultMatrix = vPhiJ.minus(vNewPointFeatures);
		double numerator = resultMatrix.transpose().times(resultMatrix).get(0, 0);
		double denominator = 2*sigmaForGaussian*sigmaForGaussian;
//		double result = Math.exp(-1*(resultMatrix.transpose().times(resultMatrix).get(0, 0))/(2*sigmaForGaussian*sigmaForGaussian));
		double result = Math.exp(-1*numerator/denominator);
//		System.out.println("kernel value: "+result);
		return result;
	}

	/**
	 * Write the learned info to a file
	 * @param mAlpha
	 * @param mLabels
	 * @param mPHI
	 * @param fileName
	 * @throws IOException
	 */
	private void writeToFile(Matrix mAlpha, Matrix mLabels, Matrix mPHI, String fileName) throws IOException {
		// Open the file to write
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));

		// File format: 
		// For each example: product of alpha & t followed by feature vectors 
		// alpha*T phi(x0) phi(x1) phi(x2) .....
		// Select Each example
		for(int i=0; i<mPHI.getColumnDimension(); i++) {
			bw.write(Double.toString((mAlpha.get(i, 0)*mLabels.get(0, i))));
			bw.write(" ");
			// Concatenate all the feature vectors for an example
			for(int j=0; j<mPHI.getRowDimension(); j++) {
				bw.write(Double.toString(mPHI.get(j ,i)));
				bw.write(" ");
			}

			//-- testing
//			double value = mAlpha.get(i, 0)*mLabels.get(0, i);
//			if(value > 0) {
//				System.out.println(i);
//			}
			//--
			// write line to file
			bw.write("\n");
		}

		// Close buffer writer
		bw.close();
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
