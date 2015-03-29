package testing;
import weka.core.matrix.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import classification.GenerateMatrices;

public class SvmConfusionMatrix {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		//		linearConfusionMatrix().print(4, 0);
//		polynomialConfusionMatrix().print(4, 0);
		//		gaussianConfusionMatrix().print(4, 0);
//		System.out.println("Linear Confusion Matrix: ");
//		confusionMatrix("_linear_svm.result").print(4, 0);
//		
//		System.out.println("Polynomial Confusion Matrix: ");
//		confusionMatrix("_polynomial5_svm.result").print(3, 0);
		
//		System.out.println("Gaussian Confusion Matrix: ");
		confusionMatrix("_svm_gaus_gamma2.result").print(3, 0);
//		confusionMatrix("_svm_gaus_gamma0.005.result").print(4, 0);
	}

	public static Matrix confusionMatrix(String str) throws IOException {

		// Get the result files of svm to read
		List<BufferedReader> fileList = new ArrayList<BufferedReader>();
		for(int i=0; i<10; i++) {
			fileList.add(new BufferedReader(new FileReader("./data/svm_results/"+ i + str)));
		}

		// Get the test file true class Matrix so that to check with predicted results
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();

		// Build the confusion matrix
		Matrix cm = new Matrix(10, 10, 0);

		// Now, each row in 10 result files give the values of discriminant functions.
		// Take the value which is maximum
		// Iterate on all the test examples
		for(int i=0; i<trueLabels.getColumnDimension(); i++) {
			// get the list of function values
			List<Double> funcValues = new ArrayList<Double>();
			for(int j=0; j<10; j++){
				funcValues.add(Double.parseDouble(fileList.get(j).readLine()));
			}

			// Now from all the values get the maximum valued class
			int systemClass = funcValues.indexOf(Collections.max(funcValues));

			// get true labels
			int trueClass = (int) trueLabels.get(0, i);

			// increment the confusion matrix
			cm.set(systemClass, trueClass, cm.get(systemClass, trueClass)+1);
		}

		// return confusion matrix
		return cm;
	}
	
/*	public static Matrix linearConfusionMatrix() throws IOException {

		// Get the result files of svm to read
		List<BufferedReader> fileList = new ArrayList<BufferedReader>();
		for(int i=0; i<10; i++) {
			fileList.add(new BufferedReader(new FileReader("./data/svm_results/"+i+"_linear_svm.result")));
		}

		// Get the test file true class Matrix so that to check with predicted results
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();

		// Build the confusion matrix
		Matrix cm = new Matrix(10, 10, 0);

		// Now, each row in 10 result files give the values of discriminant functions.
		// Take the value which is maximum
		// Iterate on all the test examples
		for(int i=0; i<trueLabels.getColumnDimension(); i++) {
			// get the list of function values
			List<Double> funcValues = new ArrayList<Double>();
			for(int j=0; j<10; j++){
				funcValues.add(Double.parseDouble(fileList.get(j).readLine()));
			}

			// Now from all the values get the maximum valued class
			int systemClass = funcValues.indexOf(Collections.max(funcValues));

			// get true labels
			int trueClass = (int) trueLabels.get(0, i);

			// increment the confusion matrix
			cm.set(systemClass, trueClass, cm.get(systemClass, trueClass)+1);
		}

		// return confusion matrix
		return cm;
	}

	public static Matrix polynomialConfusionMatrix() throws IOException {

		// Get the result files of svm to read
		List<BufferedReader> fileList = new ArrayList<BufferedReader>();
		for(int i=0; i<10; i++) {
			fileList.add(new BufferedReader(new FileReader("./data/svm_results/"+i+"_polynomial5_svm.result")));
		}

		// Get the test file true class Matrix so that to check with predicted results
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();

		// Build the confusion matrix
		Matrix cm = new Matrix(10, 10, 0);

		// Now, each row in 10 result files give the values of discriminant functions.
		// Take the value which is maximum
		// Iterate on all the test examples
		for(int i=0; i<trueLabels.getColumnDimension(); i++) {
			// get the list of function values
			List<Double> funcValues = new ArrayList<Double>();
			for(int j=0; j<10; j++){
				funcValues.add(Double.parseDouble(fileList.get(j).readLine()));
			}

			// Now from all the values get the maximum valued class
			int systemClass = funcValues.indexOf(Collections.max(funcValues));

			// get true labels
			int trueClass = (int) trueLabels.get(0, i);

			// increment the confusion matrix
			cm.set(systemClass, trueClass, cm.get(systemClass, trueClass)+1);
		}

		// return confusion matrix
		return cm;
	}

	public static Matrix gaussianConfusionMatrix() throws IOException {

		// Get the result files of svm to read
		List<BufferedReader> fileList = new ArrayList<BufferedReader>();
		for(int i=0; i<10; i++) {
			fileList.add(new BufferedReader(new FileReader("./data/svm_results/"+i+"_svm_gaus_gamma2.result")));
		}

		// Get the test file true class Matrix so that to check with predicted results
		GenerateMatrices gmTest = new GenerateMatrices("./data/optdigits.tes");
		Matrix trueLabels = gmTest.getLabelsVector();

		// Build the confusion matrix
		Matrix cm = new Matrix(10, 10, 0);

		// Now, each row in 10 result files give the values of discriminant functions.
		// Take the value which is maximum
		// Iterate on all the test examples
		for(int i=0; i<trueLabels.getColumnDimension(); i++) {
			// get the list of function values
			List<Double> funcValues = new ArrayList<Double>();
			for(int j=0; j<10; j++){
				funcValues.add(Double.parseDouble(fileList.get(j).readLine()));
			}

			// Now from all the values get the maximum valued class
			int systemClass = funcValues.indexOf(Collections.max(funcValues));

			// get true labels
			int trueClass = (int) trueLabels.get(0, i);

			// increment the confusion matrix
			cm.set(systemClass, trueClass, cm.get(systemClass, trueClass)+1);
		}

		// return confusion matrix
		return cm;
	}*/
}
