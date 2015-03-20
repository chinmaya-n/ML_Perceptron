import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.*;


public class ThreadTest {

    public static void main(final String[] argv) {
        final ExecutorService service;
        List<Future<String>>  task = new ArrayList<Future<String>>();

        service = Executors.newFixedThreadPool(10);        
        for(int i=0; i<10; i++) {
        	task.add(service.submit(new Foo(i)));
        }

        try {
            String str;

            // waits the 10 seconds for the Callable.call to finish.
            for(int j=0; j<10; j++) {
            	str = task.get(j).get();
            	System.out.println(str);
            }
        } catch(final InterruptedException ex) {
            ex.printStackTrace();
        } catch(final ExecutionException ex) {
            ex.printStackTrace();
        }

        service.shutdownNow();
    }
}

class Foo implements Callable<String> {
	private int classNo;

	public Foo(int classNo) {
		this.classNo = classNo;
	}
	
    public String call() {
        try {
            // sleep for 10 seconds
            Thread.sleep(10 * 1000);
        } catch(final InterruptedException ex) {
            ex.printStackTrace();
        }

        return Integer.toString(classNo);
    }
}