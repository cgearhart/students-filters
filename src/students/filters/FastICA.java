package students.filters;

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
	
	// The data matrix
	private SimpleMatrix X;
	
	// The estimated source matrix
	private SimpleMatrix X_;
	
	// Reference to non-linear neg-entropy estimator function
	private NegativeEntropyEstimator G;
	
	// Number of components to output
	private int num_components;
	
	// number of rows (instances) in X
	private int m;
	
	// number of columns (features) in X
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
	 * @param data - 2d array of doubles containing the source data; each column
	 * contains a single signal, and each row contains one sample of all signals 
	 * (rows contain instances, columns are features)
	 * @param g_func - {@link NegativeEntropyEstimator} to estimate negative entropy 
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 * @param whiten - whiten the data matrix (default true)
	 */
	public FastICA(NegativeEntropyEstimator g_func, double tolerance, int max_iter, boolean whiten) {
		this.G = g_func;
		this.tolerance = tolerance;
		this.max_iter = max_iter;
		this.whiten = whiten;
	}
	
	/**
	 * Default FastICA instance using the LogCosh() function to estimate
	 * negative entropy and whitening/mean-centering of the data matrix. This 
	 * implementation does not perform automatic component selection or reduction.
	 * 
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 */
	public FastICA(double tolerance, int max_iter) {
		this(new LogCosh(), tolerance, max_iter, true);
	}

	/**
	 * Estimate the unmixing matrix for the data provided
	 *
	 * @param data - 2d array of doubles containing the data; each column
	 * contains a single signal, and each row contains one sample of all 
	 * signals (rows contain instances, columns are features)
	 */
	public void fit(double[][] data) {
		this.X = center(new SimpleMatrix(data));
		this.m = X.numRows();
		this.n = X.numCols();
		
		// get the size parameter of the symmetric W matrix; size cannot be
		// larger than the number of samples or the number of features
		this.num_components = Math.min(m, n);
		
		if (this.whiten) {
			this.X = whiten(X);
		}
		
		// start with an orthogonal initial W matrix drawn from a standard Normal distribution
		W = symmetricDecorrelation(gaussianSquareMatrix(num_components));
		
		// fit the data
		parallel_ica();
	}
	
	/*
	 * FastICA main loop - default uses symmetric decorrelation. (i.e., 
	 * estimate all the independent components in parallel)
	 */
	private void parallel_ica() { 
		
		SimpleMatrix W_next, diag;
		
		for (int iter = 0; iter < max_iter; iter++) {
			
			// Calculate the neg entropy estimate and first derivative
			G.estimate(W.mult(X));
			
			// Update the W matrix
			W_next = G.getGx().transpose().mult(X).scale(1. / new Double(n));
			for (int i = 0; i < m; i++) {
				SimpleMatrix row = W_next.extractVector(true, i);
				W_next.insertIntoThis(i, 0, row.minus(row.elementMult(G.getG_x())));
			}
			W_next = symmetricDecorrelation(W_next);

			diag = W_next.mult(W.transpose()).extractDiag();
			W = W_next;
			
			// Test convergence criteria
			double largest = 0;
			for (int i = 0; i < diag.numRows(); i++) {
				double element = Math.abs(Math.abs(diag.get(i)) - 1);
				if (element > largest)
					largest = element;
			}
			if (largest < tolerance) 
				break;
		}
		
		// project the data to extract the estimated source components
		X_ = X.mult(W.mult(K));
		
		// TODO: The filter should warn before returning if the transform did not converge
	}
	
	/*
	 * Whiten a matrix of column vectors by decorrelating and scaling the 
	 * elements according to: x_new = ED^{-1/2}E'x , where E is the
	 * orthogonal matrix of eigenvectors of E{xx'}. In this implementation
	 * based on the FastICA sklearn Python package the eigen decomposition is 
	 * replaced with the SVD.
	 * 
	 * Testing this method is difficult because the decomposition is ambiguous
	 * with regard to the direction of column vectors (they can be either +/-
	 * without changing the result).
	 */
	private SimpleMatrix whiten(SimpleMatrix x) {
		// get compact SVD (D matrix is min(m,n) square)
		SimpleSVD svd = x.svd(true);
		
		// TODO: K should only keep `num_components` columns (ordered from
		// smallest to largest) if performing dimensionality reduction
		K = svd.getV().mult(svd.getW().invert());
	
		SimpleMatrix ret = x.mult(K);
		ret = ret.scale(Math.sqrt(x.numRows()));
		return ret;
	}
	
	/*
	 * Center a matrix by subtracting the average of each column vector
	 * from every element in the column
	 */
	private static SimpleMatrix center(SimpleMatrix x) {
		SimpleMatrix ret = x.copy();
		SimpleMatrix col;
		double mean;
		int m = x.numRows();
		int n = x.numCols();
		
		for (int i = 0; i < n; i++) {
			col = x.extractVector(false, i);
			mean = col.elementSum() / new Double(m);
			for (int j = 0; j < m; j++) {
				col.set(j, col.get(j) - mean);
			}
			ret.insertIntoThis(0, i, col);
		}
		return ret;
	}
	
	/*
	 * Randomly generate a square matrix drawn from a standard gaussian 
	 * distribution.
	 * 
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
	 * Perform symmetric decorrelation on the input matrix to ensure that each
	 * column is independent from all the others. This is required in order
	 * to prevent FastICA from solving for the same components in multiple
	 * columns.
	 * 
	 * NOTE: There are only real eigenvalues because x.x' is Hermitian
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
	private static SimpleMatrix symmetricDecorrelation(SimpleMatrix x) {
		SimpleEVD evd = x.mult(x.transpose()).eig();
		int len = evd.getNumberOfEigenvalues();
		SimpleMatrix u = new SimpleMatrix(len, len);
		SimpleMatrix v = new SimpleMatrix(len, len);
		double eigval;
		
		// Scale each column of the eigenvector matrix by the square root of 
		// the reciprocal of the associated eigenvalue
		for (int i = 0; i < len; i++) {
			eigval = Math.sqrt(evd.getEigenvalue(i).getReal());
			v.insertIntoThis(0, i, evd.getEigenVector(i));
			u.insertIntoThis(0, i, evd.getEigenVector(i).divide(eigval));
		}
		
		return u.mult(v.transpose()).mult(x);
	}
	
}
