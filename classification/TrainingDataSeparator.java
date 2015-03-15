package classification;

import java.io.FileReader;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.FileWriter;
import java.io.File;
import java.util.*;

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
			
			if(lineCount%500==0) {
				
			}
			// Append the line to appropriate file
			switch(sysClass) {
			case 0: 
				appendToFile(bwList, line, 0);
				break;
			case 1: 
				appendToFile(bwList, line, 1);
				break;
			case 2: 
				appendToFile(bwList, line, 2);
				break;
			case 3: 
				appendToFile(bwList, line, 3);
				break;
			case 4: 
				appendToFile(bwList, line, 4);
				break;
			case 5: 
				appendToFile(bwList, line, 5);
				break;
			case 6: 
				appendToFile(bwList, line, 6);
				break;
			case 7: 
				appendToFile(bwList, line, 7);
				break;
			case 8: 
				appendToFile(bwList, line, 8);
				break;
			case 9: 
				appendToFile(bwList, line, 9);
				break;
			default:
				System.out.println("File Class not in range of 0 to 9. Aborting!");
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
