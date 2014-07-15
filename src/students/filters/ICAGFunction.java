package students.filters;

import weka.core.matrix.Matrix;

/**
 * The G function is used as an approximation to neg-entropy in FastICA. Standard nonlinear G 
 * functions include the log-cosh (recommended as a good, general-purpose function), cube,
 * and exponential. The G function is applied element-wise to each attribute of the data as part of
 * the FastICA algorithm.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 */
public interface ICAGFunction {
	
	/**
	 * Apply the g function to the input matrix and store the of the function
	 * and its first derivative in the gx and gpx matrices respectively.
	 * 
	 * @param x - {@link Matrix} of input values
	 */
	abstract void apply(Matrix x);
	
	/**
	 * 
	 * @return {@link Matrix} containing the value of the G function applied to
	 * each value of the input matrix
	 */
	abstract Matrix getGx();
	
	/**
	 * 
	 * @return {@link Matrix} containing the value of the average of the first 
	 * derivative of the G function applied to each value of the input matrix
	 */
	abstract Matrix getGpx();

}
