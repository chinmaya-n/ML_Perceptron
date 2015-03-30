package testing;
import java.io.IOException;
//import weka.core.matrix.Matrix;
import classification.*;
public class FileTest {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		
		long totalTime = 0;
		// Train for each digit and generate a file
		for(int i=0; i<10; i++) {
			// Get the matrices
			GenerateMatrices gm = new GenerateMatrices("./data/Digit"+i+".tra");
			
			// Create kernel perceptron
			KernelPerceptron kp = new KernelPerceptron();
			kp.featureVectors = gm.getPHI();
			kp.mLabels = gm.getLabelsVector();
			kp.normalize = true;
//			kp.orderOfPolynomial = 5;
			kp.sigmaForGaussian = 0.5;

			// Train the perceptron
			kp.writeLearnedInfoToFile = "./LearnedInfo/Digit"+i+"_KernelPerceptron_Epoch5_Gaus0.5.learned";
//			kp.trainKernelPerceptron(5, Kernels.GAUSSIAN, 10);
//			long startTime = System.currentTimeMillis();
			kp.trainKernelPerceptron(5, Kernels.GAUSSIAN, 0.5);
//			kp.trainKernelPerceptron(5, Kernels.POLYNOMIAL, 5);
//			long stopTime = System.currentTimeMillis();
//			totalTime += stopTime-startTime;
			//		// Get to testing
			//		Matrix result  = kp.classify("./LearnedInfo/Digit0_POLYNOMIAL_.learned", gm.getPHI(), Kernels.POLYNOMIAL, 5);
			//		

			//		Matrix alpha = kp.trainKernelPerceptron(5, Kernels.POLYNOMIAL, 5);
			//		Matrix result = kp.classify(alpha, gm.getLabelsVector(), gm.getPHI(), gm.getPHI(), Kernels.POLYNOMIAL, 5);

			//		int count=0;
			//		for (int i=0; i<result.getColumnDimension() ;i++) {
			//			count += result.get(0, i);
			//		}
			//		result.print(1, 1);
			//		System.out.println("No of Zeros: " + (count));
		}
		
		// total time for learning all the models - disable writing to the file
		System.out.println("Total Time: "+totalTime);
	}
}
