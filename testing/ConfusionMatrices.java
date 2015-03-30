package testing;
import classification.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.*;

import weka.core.matrix.Matrix;

public class ConfusionMatrices {

	/**
	 * Generate Confusion matrices for the 6 methods 
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// perceptron confusion matrix
//		perceptronConfusionMatrix();
		
		// average perceptron confusion matrix
		avgPerceptronConfusionMatrix();
		
		// kernel perceptron - polynomial - d=5
//		kernelPerceptronPolynomialConfusionMatrix();
		
		// kernel perceptron - gaussian - sigma=0.5
//		kernelPerceptronGaussianConfusionMatrix();
		
		// average kernel perceptron - polynomial - d=5
//		avgKernelPerceptronPolynomialConfusionMatrix();
		
		// average kernel perceptron - gaussian - sigma=0.5
//		avgKernelPerceptronGaussianConfusionMatrix();
	}

	private static void perceptronConfusionMatrix() throws IOException {
		/******* Confusion Matrix for perceptron *******/
		// Max Epochs
		int maxEpochs = 5;
		Perceptron p = new Perceptron();				// perceptron object

		// Train the polynomial to get the W for each digit
		long totalTime = 0; // = System.currentTimeMillis();
		List<Matrix> wList = new ArrayList<Matrix>();	// list to store weight vector
		for(int d=0; d<10; d++) {						// d for digit
			GenerateMatrices gm = new GenerateMatrices("./data/Digit"+d+".tra", "linear");
			p.featureVectors = gm.getPHI();				// add matrix of feature vectors
			p.labelsVector = gm.getLabelsVector();		// add matrix of labels
			p.normalize = true;							// normalize the matrix
			long startTime = System.currentTimeMillis();
			wList.add(p.trainPerceptron(maxEpochs));	// appends the learned W for each digit to the list 
			long stopTime = System.currentTimeMillis();
			totalTime += stopTime-startTime;
		}
		System.out.println("Total Time: "+totalTime);
		
		//-- test
//		wList.get(0).transpose().print(4, 3);
		//

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionPerceptron = new Matrix(10, 10, 0);
		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes", "linear");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			int systemDigit = 0;			// to store the most probable digit
			double maxDiscFuncValue = p.discriminantFunction(wList.get(0), testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e));	// store the max discriminant function value

			// Get the highest score & the digit from all the digits 
			for(int d=1; d<10; d++) {		// d for digit
				double funcValue = p.discriminantFunction(wList.get(d), testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e));
				if(maxDiscFuncValue < funcValue) {
					maxDiscFuncValue = funcValue;
					systemDigit = d;
				}
			}

			// place this into the confusion matrix
			mConfusionPerceptron.set(systemDigit, (int) trueLabels.get(0, e), mConfusionPerceptron.get(systemDigit, (int) trueLabels.get(0, e))+1);
		}

		// display the matrix
		mConfusionPerceptron.print(3, 0);
	}

	private static void avgPerceptronConfusionMatrix() throws IOException {
		/******* Confusion Matrix for averaged perceptron *******/
		// Max Epochs
		int maxEpochs = 5;
		Perceptron p = new Perceptron();				// perceptron object

		// Train the polynomial to get the W for each digit
		long totalTime = 0;
		
		List<Matrix> wList = new ArrayList<Matrix>();	// list to store weight vector
		for(int d=0; d<10; d++) {						// d for digit
			GenerateMatrices gm = new GenerateMatrices("./data/Digit"+d+".tra", "linear");
			p.featureVectors = gm.getPHI();				// add matrix of feature vectors
			p.labelsVector = gm.getLabelsVector();		// add matrix of labels
			p.normalize = true;							// normalize the matrix
			long startTime = System.currentTimeMillis();
			wList.add(p.trainAveragedPerceptron(maxEpochs));	// appends the learned W for each digit to the list 
			long stopTime = System.currentTimeMillis();
			totalTime += stopTime-startTime;
		}
		System.out.println("Total Time: "+ totalTime);
		
		wList.get(0).transpose().print(4, 3);

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionPerceptron = new Matrix(10, 10, 0);
		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes", "linear");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			int systemDigit = 0;			// to store the most probable digit
			double maxDiscFuncValue = p.discriminantFunction(wList.get(0), testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e));	// store the max discriminant function value

			// Get the highest score & the digit from all the digits 
			for(int d=1; d<10; d++) {		// d for digit
				double funcValue = p.discriminantFunction(wList.get(d), testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e));
				if(maxDiscFuncValue < funcValue) {
					maxDiscFuncValue = funcValue;
					systemDigit = d;
				}
			}

			// place this into the confusion matrix
			mConfusionPerceptron.set(systemDigit, (int) trueLabels.get(0, e), mConfusionPerceptron.get(systemDigit, (int) trueLabels.get(0, e))+1);
		}

		// display the matrix
		mConfusionPerceptron.print(3, 0);
	}

	private static void kernelPerceptronPolynomialConfusionMatrix() throws IOException {
		/******* Confusion Matrix for kernel perceptron - Polynomial ^5 *******/

		KernelPerceptron kp = new KernelPerceptron();				// perceptron object
		kp.orderOfPolynomial = 5;
		kp.normalize = true;

		// As we already have trained output files i.e. ".learned" files we will use them directly
		// to predict the new labels
		List<Matrix> alphaTList = new ArrayList<Matrix>();
		Matrix mTrainingPHI = null;
		
		// Get the alpha*t for each digit And trainingPhi
		for(int d=0; d<10; d++) { // d for digit
			// Open the file to get the learned contents
			String fileName = "./LearnedInfo/Digit"+d+"_KernelPerceptron_Epoch5_Poly5.learned";
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
			// We need mPHI only once, its same in all the files
			if(d==0)
				mTrainingPHI = new Matrix(noOfFeatures, noOfExamples);	

			// Read each line and populate the data
			int lineNumber = 0;
			do {
				// Tokenize the string
				StringTokenizer stLine = new StringTokenizer(line, " ");
				// Fill the Alpha*T value in mAlphaT matrix
				mAlphaT.set(lineNumber, 0, Double.parseDouble(stLine.nextToken()));

				// We need mPHI only once, its same in all the files
				if(d==0) {	
					// Fill the feature vectors in mPHI matrix
					for(int i=0; i<noOfFeatures; i++) {
						mTrainingPHI.set(i, lineNumber, Double.parseDouble(stLine.nextToken()));
					}
				}

				// Read next line
				line = br.readLine();
				lineNumber++;
			}while(lineNumber < noOfExamples && line != null);

			// add the alpha*t to the list
			alphaTList.add(mAlphaT);
			
			// close buffer
			br.close();
		}
		// Build a dummy label list (we have already got alpha*t in a matrix)
		Matrix mTrainingLabels = new Matrix(1, mTrainingPHI.getColumnDimension(), 1);

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionKernelPerceptron = new Matrix(10, 10, 0);

		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			// Store the discriminant function value for each digit
			List<Double> discFuncValueList = new ArrayList<Double>();
			
			// Get the highest score & the digit from all the digits 
			for(int d=0; d<10; d++) {		// d for digit
				double funcValue = kp.discriminantFunction(alphaTList.get(d), mTrainingLabels, mTrainingPHI,
						testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e), Kernels.POLYNOMIAL);
				discFuncValueList.add(funcValue);
			}

			// Find the max class label out of the list
			double maxFuncValue = discFuncValueList.get(0);
			int bestClass = 0;
			for(int i=0; i<10; i++) {
				if(maxFuncValue < discFuncValueList.get(i)) {
					maxFuncValue = discFuncValueList.get(i);
					bestClass = i;
				}
			}
			
			// place this into the confusion matrix
			int trueLabel = (int) trueLabels.get(0, e);
			mConfusionKernelPerceptron.set(bestClass, trueLabel, 
					mConfusionKernelPerceptron.get(bestClass, trueLabel)+1);
		}

		// display the matrix
		mConfusionKernelPerceptron.print(3, 0);
	}
	
	private static void kernelPerceptronGaussianConfusionMatrix() throws IOException {
		/******* Confusion Matrix for kernel perceptron - Gaussian with sigma = 0.5 *******/

		KernelPerceptron kp = new KernelPerceptron();				// perceptron object
		kp.sigmaForGaussian = 0.5;

		// As we already have trained output files i.e. ".learned" files we will use them directly
		// to predict the new labels
		List<Matrix> alphaTList = new ArrayList<Matrix>();
		Matrix mTrainingPHI = null;
		
		// Get the alpha*t for each digit And trainingPhi
		for(int d=0; d<10; d++) { // d for digit
			// Open the file to get the learned contents
			String fileName = "./LearnedInfo/Digit"+d+"_KernelPerceptron_Epoch5_Gaus0.5.learned";
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
			// We need mPHI only once, its same in all the files
			if(d==0)
				mTrainingPHI = new Matrix(noOfFeatures, noOfExamples);	

			// Read each line and populate the data
			int lineNumber = 0;
			do {
				// Tokenize the string
				StringTokenizer stLine = new StringTokenizer(line, " ");
				// Fill the Alpha*T value in mAlphaT matrix
				mAlphaT.set(lineNumber, 0, Double.parseDouble(stLine.nextToken()));

				// We need mPHI only once, its same in all the files
				if(d==0) {	
					// Fill the feature vectors in mPHI matrix
					for(int i=0; i<noOfFeatures; i++) {
						mTrainingPHI.set(i, lineNumber, Double.parseDouble(stLine.nextToken()));
					}
				}

				// Read next line
				line = br.readLine();
				lineNumber++;
			}while(lineNumber < noOfExamples && line != null);

			// add the alpha*t to the list
			alphaTList.add(mAlphaT);
			
			// close buffer
			br.close();
		}
		// Build a dummy label list (we have already got alpha*t in a matrix)
		Matrix mTrainingLabels = new Matrix(1, mTrainingPHI.getColumnDimension(), 1);

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionKernelPerceptron = new Matrix(10, 10, 0);

		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			// Store the discriminant function value for each digit
			List<Double> discFuncValueList = new ArrayList<Double>();
			
			// Get the highest score & the digit from all the digits 
			for(int d=0; d<10; d++) {		// d for digit
				double funcValue = kp.discriminantFunction(alphaTList.get(d), mTrainingLabels, mTrainingPHI,
						testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e), Kernels.GAUSSIAN);
				discFuncValueList.add(funcValue);
			}

			// print the list
//			if(e==0) {
//				System.out.println(discFuncValueList);
//				System.exit(1);
//			}
			// Find the max class label out of the list
//			double maxFuncValue = discFuncValueList.get(0);
//			int bestClass = 0;
//			for(int i=0; i<10; i++) {
//				if(maxFuncValue <= discFuncValueList.get(i)) {
//					maxFuncValue = discFuncValueList.get(i);
//					bestClass = i;
//				}
//			}
			
			//Get the best class
			int bestClass = discFuncValueList.indexOf(Collections.max(discFuncValueList));
			
			// place this into the confusion matrix
			int trueLabel = (int) trueLabels.get(0, e);
			mConfusionKernelPerceptron.set(bestClass, trueLabel, 
					mConfusionKernelPerceptron.get(bestClass, trueLabel)+1);
			
//			//-- testing
//			if(bestClass!=trueLabel && bestClass==0) {
//				System.out.print(discFuncValueList);
//				System.out.println(" " + trueLabel);
//				
//				// print kernel values
//				for(int i=0; i<100; i++) {
////					System.out.print(e+" ");
//					if(e==2) {
//						System.out.println("");
//					}
//					System.out.print(" "+kp.gaussianKernel(mTrainingPHI.getMatrix(0, mTrainingPHI.getRowDimension()-1, i, i), 
//							testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e)));
//				}
//			}
//			System.out.println("");
			//--
		}

		// display the matrix
		mConfusionKernelPerceptron.print(3, 0);
	}
	
	
	private static void avgKernelPerceptronPolynomialConfusionMatrix() throws IOException {
		/******* Confusion Matrix for kernel perceptron - Polynomial ^5 *******/

		KernelPerceptron kp = new KernelPerceptron();				// perceptron object
		kp.orderOfPolynomial = 5;
		kp.normalize = true;

		// As we already have trained output files i.e. ".learned" files we will use them directly
		// to predict the new labels
		List<Matrix> alphaTList = new ArrayList<Matrix>();
		Matrix mTrainingPHI = null;
		
		// Get the alpha*t for each digit And trainingPhi
		for(int d=0; d<10; d++) { // d for digit
			// Open the file to get the learned contents
			String fileName = "./LearnedInfo/Digit"+d+"_AvgKernelPerceptron_Epoch5_Poly5.learned";
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
			// We need mPHI only once, its same in all the files
			if(d==0)
				mTrainingPHI = new Matrix(noOfFeatures, noOfExamples);	

			// Read each line and populate the data
			int lineNumber = 0;
			do {
				// Tokenize the string
				StringTokenizer stLine = new StringTokenizer(line, " ");
				// Fill the Alpha*T value in mAlphaT matrix
				mAlphaT.set(lineNumber, 0, Double.parseDouble(stLine.nextToken()));

				// We need mPHI only once, its same in all the files
				if(d==0) {	
					// Fill the feature vectors in mPHI matrix
					for(int i=0; i<noOfFeatures; i++) {
						mTrainingPHI.set(i, lineNumber, Double.parseDouble(stLine.nextToken()));
					}
				}

				// Read next line
				line = br.readLine();
				lineNumber++;
			}while(lineNumber < noOfExamples && line != null);

			// add the alpha*t to the list
			alphaTList.add(mAlphaT);
			
			// close buffer
			br.close();
		}
		// Build a dummy label list (we have already got alpha*t in a matrix)
		Matrix mTrainingLabels = new Matrix(1, mTrainingPHI.getColumnDimension(), 1);

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionKernelPerceptron = new Matrix(10, 10, 0);

		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			// Store the discriminant function value for each digit
			List<Double> discFuncValueList = new ArrayList<Double>();
			
			// Get the highest score & the digit from all the digits 
			for(int d=0; d<10; d++) {		// d for digit
				double funcValue = kp.discriminantFunction(alphaTList.get(d), mTrainingLabels, mTrainingPHI,
						testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e), Kernels.POLYNOMIAL);
				discFuncValueList.add(funcValue);
			}

			// Find the max class label out of the list
//			double maxFuncValue = discFuncValueList.get(0);
//			int bestClass = 0;
//			for(int i=0; i<10; i++) {
//				if(maxFuncValue < discFuncValueList.get(i)) {
//					maxFuncValue = discFuncValueList.get(i);
//					bestClass = i;
//				}
//			}
			// Get the best class
			int bestClass = discFuncValueList.indexOf(Collections.max(discFuncValueList));
			
			// place this into the confusion matrix
			int trueLabel = (int) trueLabels.get(0, e);
			mConfusionKernelPerceptron.set(bestClass, trueLabel, 
					mConfusionKernelPerceptron.get(bestClass, trueLabel)+1);
		}

		// display the matrix
		mConfusionKernelPerceptron.print(3, 0);
	}
	
	private static void avgKernelPerceptronGaussianConfusionMatrix() throws IOException {
		/******* Confusion Matrix for average kernel perceptron - Gaussian with sigma = 0.5 *******/

		KernelPerceptron kp = new KernelPerceptron();				// perceptron object
		kp.sigmaForGaussian = 10;

		// As we already have trained output files i.e. ".learned" files we will use them directly
		// to predict the new labels
		List<Matrix> alphaTList = new ArrayList<Matrix>();
		Matrix mTrainingPHI = null;
		
		// Get the alpha*t for each digit And trainingPhi
		for(int d=0; d<10; d++) { // d for digit
			// Open the file to get the learned contents
			String fileName = "./LearnedInfo/Digit"+d+"_AvgKernelPerceptron_Epoch5_Gaus10.learned";
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
			// We need mPHI only once, its same in all the files
			if(d==0)
				mTrainingPHI = new Matrix(noOfFeatures, noOfExamples);	

			// Read each line and populate the data
			int lineNumber = 0;
			do {
				// Tokenize the string
				StringTokenizer stLine = new StringTokenizer(line, " ");
				// Fill the Alpha*T value in mAlphaT matrix
				mAlphaT.set(lineNumber, 0, Double.parseDouble(stLine.nextToken()));

				// We need mPHI only once, its same in all the files
				if(d==0) {	
					// Fill the feature vectors in mPHI matrix
					for(int i=0; i<noOfFeatures; i++) {
						mTrainingPHI.set(i, lineNumber, Double.parseDouble(stLine.nextToken()));
					}
				}

				// Read next line
				line = br.readLine();
				lineNumber++;
			}while(lineNumber < noOfExamples && line != null);

			// add the alpha*t to the list
			alphaTList.add(mAlphaT);
			
			// close buffer
			br.close();
		}
		// Build a dummy label list (we have already got alpha*t in a matrix)
		Matrix mTrainingLabels = new Matrix(1, mTrainingPHI.getColumnDimension(), 1);

		// Build the confusion Matrix - Column: True Label; Row: Machine Label
		Matrix mConfusionKernelPerceptron = new Matrix(10, 10, 0);

		// Test on the test data 
		// Get the matrices
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();
		Matrix testPHI = gmTest.getPHI();

		// Get the perceptron
		// Test for each example
		for(int e=0; e<testPHI.getColumnDimension(); e++) {
			// Store the discriminant function value for each digit
			List<Double> discFuncValueList = new ArrayList<Double>();
			
			// Get the highest score & the digit from all the digits 
			for(int d=0; d<10; d++) {		// d for digit
				double funcValue = kp.discriminantFunction(alphaTList.get(d), mTrainingLabels, mTrainingPHI,
						testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e), Kernels.GAUSSIAN);
				discFuncValueList.add(funcValue);
			}

			// print the list
//			if(e==0) {
//				System.out.println(discFuncValueList);
//				System.exit(1);
//			}
			// Find the max class label out of the list
//			double maxFuncValue = discFuncValueList.get(0);
//			int bestClass = 0;
//			for(int i=0; i<10; i++) {
//				if(maxFuncValue <= discFuncValueList.get(i)) {
//					maxFuncValue = discFuncValueList.get(i);
//					bestClass = i;
//				}
//			}
			
			//Get the best class
			int bestClass = discFuncValueList.indexOf(Collections.max(discFuncValueList));
			
			// place this into the confusion matrix
			int trueLabel = (int) trueLabels.get(0, e);
			mConfusionKernelPerceptron.set(bestClass, trueLabel, 
					mConfusionKernelPerceptron.get(bestClass, trueLabel)+1);
			
//			//-- testing
//			if(bestClass!=trueLabel && bestClass==0) {
//				System.out.print(discFuncValueList);
//				System.out.println(" " + trueLabel);
//				
//				// print kernel values
//				for(int i=0; i<100; i++) {
////					System.out.print(e+" ");
//					if(e==2) {
//						System.out.println("");
//					}
//					System.out.print(" "+kp.gaussianKernel(mTrainingPHI.getMatrix(0, mTrainingPHI.getRowDimension()-1, i, i), 
//							testPHI.getMatrix(0, testPHI.getRowDimension()-1, e, e)));
//				}
//			}
//			System.out.println("");
			//--
		}

		// display the matrix
		mConfusionKernelPerceptron.print(3, 0);
	}
}
