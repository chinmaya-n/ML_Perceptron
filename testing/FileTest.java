package testing;
import java.io.IOException;

import weka.core.matrix.Matrix;

import classification.*;

public class FileTest {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// Get the matrices
		GenerateMatrices gm = new GenerateMatrices("./data/Digit0.dev");
		// Create kernel perceptron
		KernelPerceptron kp = new KernelPerceptron();
		kp.featureVectors = gm.getPHI();
		kp.mLabels = gm.getLabelsVector();
		
//		// Train the perceptron
//		kp.writeLearnedInfoToFile = "./LearnedInfo/Digit0_POLYNOMIAL_.learned";
//		kp.trainKernelPerceptron(5, Kernels.POLYNOMIAL, 5);
//		// Get to testing
//		Matrix result  = kp.classify("./LearnedInfo/Digit0_POLYNOMIAL_.learned", gm.getPHI(), Kernels.POLYNOMIAL, 5);
//		
		
		Matrix alpha = kp.trainKernelPerceptron(5, Kernels.POLYNOMIAL, 5);
		Matrix result = kp.classify(alpha, gm.getLabelsVector(), gm.getPHI(), gm.getPHI(), Kernels.POLYNOMIAL, 5);
		
		int count=0;
		for (int i=0; i<result.getColumnDimension() ;i++) {
			count += result.get(0, i);
		}
		result.print(1, 1);
		System.out.println("No of Zeros: " + (count));
	}
}
