package students.filters.tests;

import static org.junit.Assert.*;

import java.io.PrintWriter;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.junit.BeforeClass;
import org.junit.Test;

import students.filters.FastICA;

public class ICAtests {
	
	static SimpleMatrix signal;
	
	@BeforeClass
	public static void mixSignals() {
		int NUM_SAMPLES = 2000;
		double dt = 8 / new Double(NUM_SAMPLES);
		Random r = new Random((long) 0);
		double[] t = new double[NUM_SAMPLES];
		double[][] s = new double[NUM_SAMPLES][2];
		
		for (int i = 0; i < 2000; i++) {
			t[i] = dt * i;
			s[i][0] = Math.sin(2*t[i]) + 0.2 * r.nextDouble();
			s[i][1] = (Math.sin(3*t[i]) >= 0) ? 1 : 0;
			s[i][1] += 0.2 * r.nextDouble();
		}
		SimpleMatrix S = new SimpleMatrix(s);
		SimpleMatrix mix = new SimpleMatrix(new double[][]{{1,1}, {0.5, 2}});
		signal = S.mult(mix.transpose());
//		signal.extractMatrix(0, 3, 0, 2).print(5, 2);

		// Save the signal to a file for comparison testing with sklearn
//		try {
//			PrintWriter writer = new PrintWriter("signal.txt", "UTF-8");
//			signal.write(writer);
//		} catch (Exception e) {
//			
//		}
	}

	@Test
	public void testParallelICA() {
//		FastICA ica = new FastICA(signal, 0.01, 100);
//		Matrix W = ica.parallel_ica();
//		W.print(5, 2);
	}

}
