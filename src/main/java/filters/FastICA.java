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

import java.util.Random;

import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;



/**
 * FastICA port of Python scikit-learn implementation.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 *
 */
public class FastICA {
	
	// The estimated unmixing matrix
	private SimpleMatrix W;
	
	// The pre-whitening matrix
	private SimpleMatrix K;
	
	// the product of K and W
	private SimpleMatrix KW;
	
	// The data matrix
	private SimpleMatrix X;
	
	// The mean value of each column of the input matrix
	private SimpleMatrix X_means;
	
	// Reference to non-linear neg-entropy estimator function
	private NegativeEntropyEstimator G;
	
	// Number of components to output
	private int num_components;
	
	/** number of rows (instances) in X */
	private int m;
	
	/** number of columns (features) in X */
	private int n;
	
	// Convergence tolerance
	private final double tolerance;
	
	// Iteration limit
	private final int max_iter;
	
	// Whiten the data if true
	private final boolean whiten;
	
	/**
	 * General FastICA instance constructor with an arbitrary (user-supplied)
	 * function to estimate negative entropy. This implementation does not
	 * perform automatic component selection or reduction.
	 * 
	 * @param	data 		2d array of doubles containing the source data; 
	 * 				each column contains a single signal, and each row contains
	 * 				one sample of all signals (rows contain instances, 
	 * 				columns are features)
	 * @param	g_func		{@link NegativeEntropyEstimator} to estimate 
	 * 				negative entropy 
	 * @param	tolerance 	maximum allowable convergence error
	 * @param	max_iter 	max number of iterations
	 * @param	whiten 		whiten the data matrix (default true)
	 */
	public FastICA(NegativeEntropyEstimator g_func, 
			double tolerance, int max_iter, boolean whiten) {
		this.G = g_func;
		this.tolerance = tolerance;
		this.max_iter = max_iter;
		this.whiten = whiten;
	}
	
	/**
	 * FastICA instance using the LogCosh() function to estimate negative 
	 * entropy and mean-centering with optional whitening of the data matrix.
	 * This implementation does not perform automatic component selection or 
	 * reduction.
	 * 
	 * @param	tolerance	maximum allowable convergence error
	 * @param	max_iter	max number of iterations
	 * @param	whiten		whiten the data matrix (default true) 
	 */
	public FastICA(double tolerance, int max_iter, boolean whiten) {
		this(new LogCosh(), tolerance, max_iter, whiten);
	}
	
	/**
	 * FastICA instance using the LogCosh() function to estimate negative 
	 * entropy and whitening/mean-centering of the data matrix. This 
	 * implementation does not perform automatic component selection or 
	 * reduction.
	 * 
	 * @param	tolerance 	maximum allowable convergence error
	 * @param	max_iter 	max number of iterations
	 */
	public FastICA(double tolerance, int max_iter) {
		this(new LogCosh(), tolerance, max_iter, true);
	}
	
	/**
	 * Default FastICA instance using the LogCosh() function to estimate
	 * negative entropy and whitening/mean-centering of the data matrix with
	 * simple default values.
	 */
	public FastICA() {
		this(new LogCosh(), 1E-5, 200, true);
	}
	
	/**
	 * Return the matrix that projects the data into the ICA domain
	 * 
	 * @return 	W	double[][] matrix containing estimated independent component 
	 * 			projection matrix
	 */
	public double[][] getW() {
		return FastICA.mToA(W);
	}
	
	/**
	 * Return the pre-whitening matrix that was used on the data (defaults to
	 * the identity matrix)
	 * 
	 * @return	K 	double[][] matrix containing the pre-whitening matrix
	 */
	public double[][] getK() {
		return FastICA.mToA(K);
	}
	
	/**
	 * Return the estimated mixing matrix that maps sources to the data domain
	 * 
	 * S * em = X
	 * 
	 * @return	em	double[][] matrix containing the estimated mixing matrix 
	 */
	public double[][] getEM() {
		return FastICA.mToA(K.mult(W).pseudoInverse());
	}
	
	/**
	 * Project a row-indexed matrix of data into the ICA domain by applying
	 * the pre-whitening and un-mixing matrices. This method should not be
	 * called prior to running fit() with input data.
	 * 
	 * @param 	data	rectangular double[][] array containing values; the
	 * 				number of columns should match the data provided to the
	 * 				fit() method for training 
	 * @return	result	rectangular double[][] array containing the projected
	 * 				output values
	 */
	public double[][] transform(double[][] data) {
		SimpleMatrix x = new SimpleMatrix(data);
		return FastICA.mToA(x.minus(X_means).mult(KW));
	}

	/**
	 * Estimate the unmixing matrix for the data provided
	 *
	 * @param data - 2d array of doubles containing the data; each column
	 * contains a single signal, and each row contains one sample of all 
	 * signals (rows contain instances, columns are features)
	 */
	public void fit(double[][] data, int num_components) throws Exception {
		X = new SimpleMatrix(data);
		m = X.numRows();
		n = X.numCols();

		// mean center the attributes in X
		double[] means = center(X);
		X_means = new SimpleMatrix(new double[][]{means});
		
		// get the size parameter of the symmetric W matrix; size cannot be
		// larger than the number of samples or the number of features
		this.num_components = Math.min(Math.min(m, n), num_components);
		
		K = SimpleMatrix.identity(this.num_components);  // init K
		if (this.whiten) {
			X = whiten(X);  // sets K
		}
				
		// start with an orthogonal initial W matrix drawn from a standard Normal distribution
		W = symmetricDecorrelation(gaussianSquareMatrix(num_components));

		// fit the data
		parallel_ica();  // solves for W
		
		// Store the resulting transformation matrix
		KW = K.mult(W);

	}
	
	/*
	 * FastICA main loop - using default symmetric decorrelation. (i.e., 
	 * estimate all the independent components in parallel)
	 */
	private void parallel_ica() throws Exception {
		
		double tmp;
		double lim;
		SimpleMatrix W_next;
		SimpleMatrix newRow;
		SimpleMatrix oldRow;
		
		for (int iter = 0; iter < max_iter; iter++) {
			
			// Estimate the negative entropy and first derivative average
			G.estimate(X.mult(W));
			
			// Update the W matrix
			W_next = X.transpose().mult(G.getGx()).scale(1. / new Double(n));			
			for (int i = 0; i < num_components; i++) {
				newRow = W_next.extractVector(true, i);
				oldRow = W.extractVector(true, i);
				W_next.insertIntoThis(i, 0, 
						newRow.minus(oldRow.elementMult(G.getG_x())));
			}
			W_next = symmetricDecorrelation(W_next);

			// Test convergence criteria for W
			lim = 0;
			for (int i = 0; i < W.numRows(); i++) {
				newRow = W_next.extractVector(true, i);
				oldRow = W.extractVector(true, i);
				tmp = newRow.dot(oldRow.transpose());
				tmp = Math.abs(Math.abs(tmp) - 1);
				if (tmp > lim) {
					lim = tmp;
				}
			}
			W = W_next;
			
			if (lim < tolerance) {
				return;
			}
		}
		
		throw new Exception("ICA did not converge - try again with more iterations.");
	}
	
	/*
	 * Whiten a matrix of column vectors by decorrelating and scaling the 
	 * elements according to: x_new = ED^{-1/2}E'x , where E is the
	 * orthogonal matrix of eigenvectors of E{xx'}. In this implementation
	 * (based on the FastICA sklearn Python package) the eigen decomposition is 
	 * replaced with the SVD.
	 * 
	 * The decomposition is ambiguous with regard to the direction of 
	 * column vectors (they can be either +/- without changing the result).
	 */
	@SuppressWarnings("rawtypes")
	private SimpleMatrix whiten(SimpleMatrix x) {
		// get compact SVD (D matrix is min(m,n) square)
		SimpleSVD svd = x.svd(true);
		
		// K should only keep `num_components` columns if performing 
		// dimensionality reduction
		K = svd.getV().mult(svd.getW().invert())
				.extractMatrix(0, x.numCols(), 0, num_components);
//		K = K.scale(-1);  // sklearn returns this version for K; doesn't affect results
		
//		return x.mult(K).scale(Math.sqrt(m));  // sklearn scales the input
		return x.mult(K);
	}
	
	/*
	 * Center the input matrix and store it in X by subtracting the average of 
	 * each column vector from every element in the column
	 */
	private double[] center(SimpleMatrix x) {
		SimpleMatrix col;
		int numrows = x.numRows();
		int numcols = x.numCols();
		double[] means;
		
		means = new double[numcols];
		for (int i = 0; i < numcols; i++) {
			col = x.extractVector(false, i);
			means[i] = col.elementSum() / new Double(numrows);
			for (int j = 0; j < numrows; j++) {
				col.set(j, col.get(j) - means[i]);
			}
			X.insertIntoThis(0, i, col);
		}
		
		return means;
	}
	
	/*
	 * Perform symmetric decorrelation on the input matrix to ensure that each
	 * column is independent from all the others. This is required in order
	 * to prevent FastICA from solving for the same components in multiple
	 * columns.
	 * 
	 * NOTE: There are only real eigenvalues for the W matrix
	 * 
	 * W <- W * (W.T * W)^{-1/2}
	 * 
	 * Python (Numpy):
	 *   s, u = linalg.eigh(np.dot(W.T, W)) 
	 *   W = np.dot(W, np.dot(u * (1. / np.sqrt(s)), u))
	 * Matlab: 
	 *   B = B * real(inv(B' * B)^(1/2))
	 * 
	 */
	@SuppressWarnings("rawtypes")
	private static SimpleMatrix symmetricDecorrelation(SimpleMatrix x) {
		
		double d;
		SimpleMatrix QL;
		SimpleMatrix Q;
		
		SimpleEVD evd = x.transpose().mult(x).eig();
		int len = evd.getNumberOfEigenvalues();
		QL = new SimpleMatrix(len, len);
		Q = new SimpleMatrix(len, len);
		
		// Scale each column of the eigenvector matrix by the square root of 
		// the reciprocal of the associated eigenvalue
		for (int i = 0; i < len; i++) {
			d = evd.getEigenvalue(i).getReal();
			d = (d + Math.abs(d)) / 2;  // improve numerical stability by eliminating small negatives near singular matrix zeros
			QL.insertIntoThis(0, i, evd.getEigenVector(i).divide(Math.sqrt(d)));
			Q.insertIntoThis(0, i, evd.getEigenVector(i));
		}
		
		return x.mult(QL.mult(Q.transpose()));
	}
	
	/*
	 * Randomly generate a square matrix drawn from a standard gaussian 
	 * distribution.
	 */
	private static SimpleMatrix gaussianSquareMatrix(int size) {
		SimpleMatrix ret = new SimpleMatrix(size, size);
		Random rand = new Random();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				ret.set(i, j, rand.nextGaussian());
			}
		}
		return ret;
	}
	
	/*
	 * Convert a {@link SimpleMatrix} to a 2d array of double[][]
	 */
	private static double[][] mToA(SimpleMatrix x) {
		double[][] result = new double[x.numRows()][];
		for (int i = 0; i < x.numRows(); i++) {
			result[i] = new double[x.numCols()];
			for (int j = 0; j < x.numCols(); j++) {
				result[i][j] = x.get(i, j);
			}
		}
		return result;
	}
	
}
