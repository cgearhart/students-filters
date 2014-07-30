package students.filters;

import org.ejml.simple.SimpleMatrix;


/**
 * The entropy function is an approximation to neg-entropy in FastICA. Commonly used nonlinear 
 * functions include log-cosh (recommended as a good, general-purpose function), cubic,
 * and exponential families. The function is applied element-wise to each attribute of the data
 * as part of the FastICA algorithm.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 */
public interface NegativeEntropyEstimator {
	
	/**
	 * Estimate the negative entropy of the input data matrix and store the results.
	 * 
	 * @param x {@link SimpleMatrix} containing column vectors of data to transform
	 */
	abstract void estimate(SimpleMatrix x);
	
	/**
	 * 
	 * @return {@link SimpleMatrix} containing the value of the G function applied to
	 * each value of the input matrix
	 */
	abstract SimpleMatrix getGx();
	
	/**
	 * 
	 * @return {@link SimpleMatrix} containing the value of the average of the first 
	 * derivative of the G function applied to each value of the input matrix
	 */
	abstract SimpleMatrix getG_x();

}
