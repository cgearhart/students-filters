package students.filters;

import weka.core.matrix.Matrix;

/**
 * Default G function used in FastICA to approxmate neg-entropy.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 *
 */
public class LogCosh implements ICAGFunction {
	
	private Matrix gx;
	
	private Matrix g_x;
	
	private double alpha;
	
	public LogCosh(double alpha) {
		this.alpha = alpha;
	}
	
	public LogCosh() {
		this(1.);
	}
	
	public void apply(Matrix x) {
		int rows = x.getRowDimension();
		int cols = x.getColumnDimension();
		gx = new Matrix(rows, cols);
		g_x = new Matrix(rows, 1);
		double[][] gx_ = gx.getArray();
		double[][] g_x_ = g_x.getArray();	
		double[][] x_ = x.getArray();
		
		for (int i = 0; i < rows; i++) {
			double mean = 0;
			for (int j = 0; j < cols; j++) {
				gx_[i][j] = Math.tanh(x_[i][j]);
				mean += alpha * (1 - Math.pow(gx_[i][j], 2));
			}
			g_x_[i][0] = mean / new Double(cols);
		}
	}
	
	public Matrix getGx() {
		return gx;
	}
	
	public Matrix getGpx() {
		return g_x;
	}
}
