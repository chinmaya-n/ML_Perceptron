package classification;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.StringTokenizer;

import weka.core.matrix.Matrix;

public class GenerateMatrices {

	// PHI matrix - contains all the feature vectors for the given data
	private Matrix phi;
	// labels matrix - contains all the class labels for the given data
	private Matrix vLabels;
	
	/**
	 * Constructor. Takes input as file & Generates Matrices
	 * @param dataFile
	 * @throws IOException
	 */
	public GenerateMatrices(String dataFile) throws IOException {

		// Get the number of lines the data file -> no of examples ( used in PHI, T matrix dimension )
		int noOfExamples = 0;
		LineNumberReader lnr = new LineNumberReader(new FileReader(dataFile));
		while(lnr.readLine() != null) {
			noOfExamples++;
		}
		lnr.close();

		// Read the data file
		BufferedReader br = new BufferedReader(new FileReader(dataFile));
		String line = br.readLine();

		// Count no of features in each of the feature vectors. -> ( used in PHI matrix dimension )
		StringTokenizer tokenizer = new StringTokenizer(line, ",");
		int noOfFeatures = tokenizer.countTokens()-1;

		// Create PHI matrix with dimensions: noOfFeatures X noOfExamples
		phi = new Matrix(noOfFeatures, noOfExamples);

		// Create T matrix which is class classifier matrix. Dim: 1 x noOfExamples
		vLabels = new Matrix(1, noOfExamples);

		// Fill the Matrix with given elements i.e noOFExamples
		int e=0;
		do {
			// tokenize the given string with delimiter ',' for each new line in iteration
			tokenizer = new StringTokenizer(line, ",");

			// Fill the PHI matrix i.e. feature vectors matrix
			for(int f=0; f<noOfFeatures; f++) {
				phi.set(f, e, Double.parseDouble(tokenizer.nextElement().toString()));
			}
			// Fill the label matrix with the last token i.e. class identifier
			vLabels.set(0, e, Double.parseDouble(tokenizer.nextElement().toString()));

			// move to next training example
			e++;
			line = br.readLine();
		} while(line != null && e<noOfExamples);

		// Close
		br.close();
	}
	
	/**
	 * Constructor. Takes input as file & Generates Matrices
	 * @param dataFile
	 * @throws IOException
	 */
	public GenerateMatrices(String dataFile, String perceptronType) throws IOException {

		// Check if matrices are for linear perceptron
		if(perceptronType == "linear") {
			
			// Get the number of lines the data file -> no of examples ( used in PHI, T matrix dimension )
			int noOfExamples = 0;
			LineNumberReader lnr = new LineNumberReader(new FileReader(dataFile));
			while(lnr.readLine() != null) {
				noOfExamples++;
			}
			lnr.close();

			// Read the data file
			BufferedReader br = new BufferedReader(new FileReader(dataFile));
			String line = br.readLine();

			// Count no of features in each of the feature vectors. -> ( used in PHI matrix dimension )
			StringTokenizer tokenizer = new StringTokenizer(line, ",");
			int noOfFeatures = tokenizer.countTokens()-1;
			
			// add x0 feature as well into the features (linear perceptron)
			noOfFeatures+=1;

			// Create PHI matrix with dimensions: noOfFeatures X noOfExamples
			phi = new Matrix(noOfFeatures, noOfExamples);

			// Create T matrix which is class classifier matrix. Dim: 1 x noOfExamples
			vLabels = new Matrix(1, noOfExamples);

			// Fill the Matrix with given elements i.e noOFExamples
			int e=0;
			do {
				// tokenize the given string with delimiter ',' for each new line in iteration
				tokenizer = new StringTokenizer(line, ",");

				// Fill the first value x0 = 1 for all the examples
				phi.set(0, e, 1);
				
				// Fill the PHI matrix i.e. feature vectors matrix
				for(int f=1; f<noOfFeatures; f++) {
					phi.set(f, e, Double.parseDouble(tokenizer.nextElement().toString()));
				}
				// Fill the label matrix with the last token i.e. class identifier
				vLabels.set(0, e, Double.parseDouble(tokenizer.nextElement().toString()));

				// move to next training example
				e++;
				line = br.readLine();
			} while(line != null && e<noOfExamples);

			// Close
			br.close();
		}
		else {
			System.out.println("Please use the single parameter constructor. This is exclusive for linear perceptron as need to add w0 & x0");
		}
	}
	
	/**
	 * Returns PHI matrix
	 * @return phi matrix
	 */
	public Matrix getPHI() {
		return phi;
	}
	
	/**
	 * Returns vLabels matrix
	 * @return labels matrix
	 */
	public Matrix getLabelsVector() {
		return vLabels;
	}
}
