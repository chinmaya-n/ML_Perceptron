package testing;

import classification.*;
import java.io.IOException;
import weka.core.matrix.Matrix;

public class Test {

	/**
	 * Test Program for all the perceptrons
	 * @param argv : argv[0] = training data file; argv[1] = test data file;
	 * 				argv[2] = perceptron type i.e 1-simple; 2-Avg; 3-Kernel; 4-Avg Kernel
	 * 				argv[3] = max Epoch count;	
	 * 				argv[4] = kernel type i.e 1-linear; 2-quadratic; 3-polynomial; 4-gaussian
	 * 				argv[5] = degree/sigma for polynomial/gaussian
	 * @throws IOException
	 */
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
		{	
			// Get the linear perceptron matrix values
			gmTraining = new GenerateMatrices(trainingDataFile, "linear");
			mPHI = gmTraining.getPHI();
			mLabels = gmTraining.getLabelsVector();
			
			// Create perceptron object
			Perceptron perceptron = new Perceptron();
			perceptron.featureVectors = mPHI;
			perceptron.labelsVector = mLabels;
			perceptron.normalize = true;
			
			// Send the data for training simple perceptron & get W
			Matrix mW = perceptron.trainPerceptron(maxEpochs);
			mW.transpose().print(3, 2);
			// Send for testing using the learned weights vector
			//
			
			// break
			break;
		}
		case 2: // Average Perceptron
		{	
			// Get the linear perceptron matrix values
			gmTraining = new GenerateMatrices(trainingDataFile, "linear");
			mPHI = gmTraining.getPHI();
			mLabels = gmTraining.getLabelsVector();
			
			// Create perceptron object
			Perceptron perceptron = new Perceptron();
			perceptron.featureVectors = mPHI;
			perceptron.labelsVector = mLabels;
			
			// Send the data for training simple perceptron & get W
			Matrix mW = perceptron.trainAveragedPerceptron(maxEpochs);
			mW.print(3, 2);
			// Send for testing using the learned weights vector
			//
			
			// break
			break;
		}
		case 3:	// Kernel Perceptron
		{	
			// Create Kernel Perceptron object
			KernelPerceptron kernelPerceptron = new KernelPerceptron();
			kernelPerceptron.featureVectors = mPHI;
			kernelPerceptron.mLabels = mLabels;
			kernelPerceptron.normalize = true;
			
			// What kind of kernel to be used?
			int kType = Integer.parseInt(argv[4]);
			
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
			}
			else if(kType==4) {
				kEnumType = Kernels.GAUSSIAN;
			}
			else {
				System.out.println("Kernel Type number must be from these four. \n 1. Linear\n 2. Quadratic\n 3. Polynomial\n 4. Gaussian");
				System.exit(-1);
			}
			
			// Write to file
//			kernelPerceptron.writeLearnedInfoToFile = "./LearnedInfo/Kernel_" + kEnumType.toString() + "_" + argv[5] + ".learned";
			
			// Send the data for training & get alpha
			Matrix mAlpha = kernelPerceptron.trainKernelPerceptron(maxEpochs, kEnumType, Double.parseDouble(argv[5]));
			mAlpha.print(1, 0);
			// Send for testing using the mAlpha
//			kernelPerceptron.classify(mAlpha, mLabels, mPHI, testPHI, kEnumType, Double.parseDouble(argv[5])).print(1, 1);
			
			// break
			break;
		}
		case 4:	// Average Kernel Perceptron
		{	
			// Create Kernel Perceptron object
			KernelPerceptron kernelPerceptron = new KernelPerceptron();
			kernelPerceptron.featureVectors = mPHI;
			kernelPerceptron.mLabels = mLabels;
			
			// What kind of kernel to be used?
			int kType = Integer.parseInt(argv[4]);
			
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
			}
			else if(kType==4) {
				kEnumType = Kernels.GAUSSIAN;
			}
			else {
				System.out.println("Kernel Type number must be from these four. \n 1. Linear\n 2. Quadratic\n 3. Polynomial\n 4. Gaussian");
				System.exit(-1);
			}
			
			// Send the data for training & get alpha
			Matrix mAvgAlpha = kernelPerceptron.trainAveragedKernelPerceptron(maxEpochs, kEnumType, Double.parseDouble(argv[5]));
			// Send for testing using the mAlpha
//			kernelPerceptron.classify(mAvgAlpha, mLabels, mPHI, testPHI, kEnumType).print(1, 1);
			
			// break
			break;
		}
		default:
			System.out.println("Select from the 4 types of perceptrons. \n 1. Simple Perceptron\n " +
					"2. Averaged Perceptron\n 3. Kernel Perceptron\n 4. Averaged Kernel Perceptron");
		}
	}
}
