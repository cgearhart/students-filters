/* 
 * This is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <http://unlicense.org/>
 */

package filters;

import org.ejml.simple.SimpleMatrix;


/**
 * The entropy function is an approximation to neg-entropy in FastICA. Commonly 
 * used nonlinear functions include log-cosh (recommended as a good, 
 * general-purpose function), cubic, and exponential families. The function is
 * applied element-wise to each attribute of the data as part of the FastICA 
 * algorithm.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 */
public interface NegativeEntropyEstimator {
	
	/**
	 * Estimate the negative entropy of the input data matrix and store 
	 * the results.
	 * 
	 * @param 	x 		{@link SimpleMatrix} containing column vectors of data 
	 * 				to transform
	 */
	abstract void estimate(SimpleMatrix x);
	
	/**
	 * 
	 * @return 	{@link SimpleMatrix} 	containing the value of the G function 
	 * 				applied to each value of the input matrix
	 */
	abstract SimpleMatrix getGx();
	
	/**
	 * 
	 * @return 	{@link SimpleMatrix} 	containing the value of the average of
	 * 				the first derivative of the G function applied to each 
	 * 				value of the input matrix
	 */
	abstract SimpleMatrix getG_x();

}
