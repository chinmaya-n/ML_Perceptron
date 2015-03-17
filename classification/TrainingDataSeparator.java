package classification;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.FileWriter;
import java.io.File;
import java.util.*;

/**
 * 
 * @author cn262114
 *
 */
public class TrainingDataSeparator {
	
	/**
	 * Takes the training data file which has all the examples for different classes.
	 * Then generates training file for each class making +1 for its own class example
	 * and -1 for all other examples.
	 * @param argv - argv[0] = path to the training file
	 * @throws IOException
	 */
	public static void main(String[] argv) throws IOException {
		// Training file
		String trainDataFile = argv[0];
		
		// Open the given file for reading
		BufferedReader trainData = new BufferedReader(new FileReader(trainDataFile));
		
		// Build a list of files to write the training data into them, based on their class
		List<File> fileList = new ArrayList<File>();
		for(int i=0; i<10; i++) {
			fileList.add(new File("Digit"+i+".tra"));	// .dev for development files - .tra for training files
		}
		// Create a list of write buffers to write into these files
		List<BufferedWriter> bwList = new ArrayList<BufferedWriter>();
		for(int i=0; i<10; i++) {
			bwList.add(new BufferedWriter(new FileWriter(fileList.get(i))));
		}
		
		// Now read each line from the training file and add it to appropriate data file
		String line = trainData.readLine();
		int lineCount = 0;
		while(line != null) {
			// Get the line length
			int lineLength = line.length();
			System.out.println(lineLength);
			lineCount++;
			
			// Get the class of the point
			int sysClass = Integer.parseInt(Character.toString(line.charAt(lineLength-1)));
			
			// Append the line to appropriate file
			if(sysClass<=9 && sysClass>=0) {
				appendToFile(bwList, line, sysClass);
			}
			else {
				System.out.println("Class range should be from 0 to 9. Check at line: " + lineCount);
				System.exit(-1);
			}
			
			line = trainData.readLine();
		}
		
		// Close the files
		trainData.close();
		for(int i=0; i<10; i++) {
			bwList.get(i).close();
		}
	}
	
	private static void appendToFile(List<BufferedWriter> bwList, String line, int sysClass) throws IOException {
		for(int i=0; i<10; i++) {
			if(i==sysClass) {
				bwList.get(i).write(line.substring(0, line.length()-1) + "+1" + "\n");
			}
			else {
				bwList.get(i).write(line.substring(0, line.length()-1) + "-1" + "\n");
			}
		}
	}
}
