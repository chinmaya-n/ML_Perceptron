package testing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.StringTokenizer;

public class SvmFileGeneration {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub

		for(int j=0; j<10; j++) {
			// Read the file
			BufferedReader br = new BufferedReader(new FileReader("./data/Digit"+j+".tes"));

			// open a file to write
			BufferedWriter bw = new BufferedWriter(new FileWriter("./data/Digit"+j+"_svm.tes"));

			// read each line from br and convert it to bw
			String line = br.readLine();
			StringTokenizer tokenizer;

			while(line != null) {
				// tokenize the string
				tokenizer = new StringTokenizer(line, ",");

				// total count
				int totalTokents = tokenizer.countTokens();

				// feature string
				String featureString = "";

				// write all the features which are not zeros featureNo:feature_value
				for(int i=1; i<totalTokents; i++) {
					Double featureValue = Double.parseDouble(tokenizer.nextToken());
					if(featureValue != 0) {
						featureString += " "+i+":"+featureValue.toString();
					}
				}

				String finalString = tokenizer.nextToken() + featureString;

				// write to file
				bw.write(finalString);

				// add new line to file
				bw.write("\n");

				// read next line
				line = br.readLine();
			}
			
			// print
			System.out.println("Generated file: "+j);

			// close buffers
			br.close();
			bw.close();
		}
	}

}
