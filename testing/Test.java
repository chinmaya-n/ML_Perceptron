package testing;

import classification.*;
import java.io.IOException;
import weka.core.matrix.Matrix;

public class Test {

	public static void main(String[] argv) throws IOException {
		
		// Get Training File
		String trainingDataFile = argv[0];
		// Get Test Data file
		String testDataFile = argv[1];
		// What kind of perceptron has to be used?
		int pType = Integer.parseInt(argv[2]);
		// Get Max No of epochs to run
		int maxEpochs = Integer.parseInt(argv[3]);

		// Get the matrices from the given training file
		GenerateMatrices gmTraining = new GenerateMatrices(trainingDataFile);
		Matrix mPHI = gmTraining.getPHI();
		Matrix mLabels = gmTraining.getLabelsVector();
		
		// Get the matrices from the given testing file
		GenerateMatrices gmTesting = new GenerateMatrices(testDataFile);
		Matrix testPHI = gmTesting.getPHI();
		Matrix trueTestLabels = gmTesting.getLabelsVector();

		// Get to the appropriate perceptron
		switch(pType) {
		case 1: // Simple Perceptron
		{	// Send the data for training simple perceptron & get W
			Matrix mW = Perceptron.trainPerceptron(mPHI, mLabels, maxEpochs);
			// If the data is not converged mW will be a single element matrix. Test it
			if(mW.getRowDimension()+mW.getColumnDimension() != 2) {
				// Send data for testing now!

			}
			// break
			break;
		}
		case 2: // Average Perceptron
		{	// Send the data for training simple perceptron & get W
			Matrix mW = Perceptron.trainAveragedPerceptron(mPHI, mLabels, maxEpochs);
			// If the data is not converged mW will be a single element matrix. Test it
			if(mW.getRowDimension()+mW.getColumnDimension() != 2) {
				// Send data for testing now!
				Perceptron.classify(mW, testPHI).print(1, 1);
			}
			// break
			break;
		}
		case 3:	// Kernel Perceptron
		{	// What kind of kernel to be used?
			int kType = Integer.parseInt(argv[4]);
			
			// For the value for 'M' if its polynomial kernel or 'sigma' if Gaussian kernel 
			double value = 0;
			// Convert kernel int to Kernels enum
			Kernels kEnumType = null;
			if(kType==1) {
				kEnumType = Kernels.LINEAR;
			}
			else if(kType==2) {
				kEnumType = Kernels.QUADRATIC;
			}
			else if(kType==3) {
				kEnumType = Kernels.POLYNOMIAL;
				// What is the value for M for polynomial kernel 
				value = Double.parseDouble(argv[5]);
				KernelPerceptron.orderOfPolynomial = (int) value;
			}
			else if(kType==4) {
				kEnumType = Kernels.GAUSSIAN;
				// Value of sigma if Gaussian kernel
				value = Double.parseDouble(argv[5]);
				KernelPerceptron.sigmaForGaussian = value;
			}
			else {
				System.out.println("Kernel Type number must be from these four. \n 1. Linear\n 2. Quadratic\n 3. Polynomial\n 4. Gaussian");
				System.exit(-1);
			}
			
			// Send the data for training & get alpha
			Matrix mAlpha = KernelPerceptron.trainKernelPerceptron(mPHI, mLabels, maxEpochs, kEnumType);
			// Check if the data is converged or not
			// If the data is not converged 1st value in mAlpha matrix would be -1
			if(mAlpha.get(0,0) != -1) {
				// Send data for testing now!
				
			}
			
			// break
			break;
		}
		case 4:	// Average Kernel Perceptron
		{	// What kind of kernel to be used?
			int kType = Integer.parseInt(argv[4]);
			
			// For the value for 'M' if its polynomial kernel or 'sigma' if Gaussian kernel 
			double value = 0;
			// Convert kernel int to Kernels enum
			Kernels kEnumType = null;
			if(kType==1) {
				kEnumType = Kernels.LINEAR;
			}
			else if(kType==2) {
				kEnumType = Kernels.QUADRATIC;
			}
			else if(kType==3) {
				kEnumType = Kernels.POLYNOMIAL;
				// What is the value for M for polynomial kernel 
				value = Double.parseDouble(argv[5]);
				KernelPerceptron.orderOfPolynomial = (int) value;
			}
			else if(kType==4) {
				kEnumType = Kernels.GAUSSIAN;
				// Value of sigma if Gaussian kernel
				value = Double.parseDouble(argv[5]);
				KernelPerceptron.sigmaForGaussian = value;
			}
			else {
				System.out.println("Kernel Type number must be from these four. \n 1. Linear\n 2. Quadratic\n 3. Polynomial\n 4. Gaussian");
				System.exit(-1);
			}
			
			// Send the data for training & get alpha
			Matrix mAvgAlpha = KernelPerceptron.trainAveragedKernelPerceptron(mPHI, mLabels, maxEpochs, kEnumType);
			// Check if the data is converged or not
			// If the data is not converged 1st value in mAlpha matrix would be -1
			if(mAvgAlpha.get(0,0) != -1) {
				// Send data for testing now!
				
			}
			
			// break
			break;
		}
		default:
			System.out.println("Select from the 4 types of perceptrons. \n 1. Simple Perceptron\n " +
					"2. Averaged Perceptron\n 3. Kernel Perceptron\n 4. Averaged Kernel Perceptron");
		}
	}
}
