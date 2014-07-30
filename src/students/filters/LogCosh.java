package students.filters;

import org.ejml.simple.SimpleMatrix;

/**
 * Default function used in FastICA to approxmate neg-entropy.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 *
 */
public class LogCosh implements NegativeEntropyEstimator {
	
	// Element-wise application of the neg-entropy function applied to data matrix
	private SimpleMatrix gx;
	
	// Column-wise average of the first derivative of gx; i.e., the average of 1 - gx[i]**2
	private SimpleMatrix g_x;
	
	// Scaling factor
	private final double alpha;
	
	public LogCosh(double alpha) {
		this.alpha = alpha;
	}
	
	public LogCosh() {
		this(1.);
	}
	
	/**
	 * 
	 * @param x - {@link SimpleMatrix} of column vectors for each feature
	 */
	@Override
	public void estimate(SimpleMatrix x) {
		int m = x.numRows();
		int n = x.numCols();
		gx = new SimpleMatrix(m, n);
		g_x = new SimpleMatrix(1, n);
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				gx.set(i, j, Math.tanh(x.get(i, j)));
				double gx_i = alpha * (1 - Math.pow(gx.get(i, j), 2));
				g_x.set(0, j, g_x.get(i,j) + gx_i);
			}
			g_x.set(0, j, g_x.get(0, j) / new Double(m));
		}

	}
	
	@Override
	public SimpleMatrix getGx() {
		return gx;
	}
	
	@Override
	public SimpleMatrix getG_x() {
		return g_x;
	}
}
