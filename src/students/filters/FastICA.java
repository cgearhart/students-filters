package students.filters;

import java.util.Arrays;
import java.util.Random;

import weka.core.matrix.EigenvalueDecomposition;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;



/**
 * FastICA port from scikit-learn implementation.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 *
 */
public class FastICA {
	
	// 
	private Matrix S;
	
	// The data matrix
	private Matrix X;
	
	// The unmixing matrix
	private Matrix W;
	
	// Reference to non-linear neg-entropy estimator function
	private ICAGFunction G;
	
	// Convergence tolerance
	private final float tolerance;
	
	// Iteration limit
	private final int max_iter;
	
	// number of components to output
	private final int num_components;
	
	/**
	 * General FastICA instance constructor with an arbitrary (user-supplied)
	 * function to estimate negative entropy. This implementation does not
	 * perform automatic component selection or reduction.
	 * 
	 * @param data - {@link Matrix} containing the source data; each column
	 * contains a single signal, and each row contains one sample of all signals 
	 * (rows contain instances, columns are features)
	 * @param g_func - {@link ICAGFunction} to estimate negative entropy 
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 * @param whiten - whiten the data matrix (default true)
	 */
	public FastICA(Matrix data, ICAGFunction g_func, float tolerance, int max_iter, boolean whiten) {
		this.G = g_func;
		this.tolerance = tolerance;
		this.max_iter = max_iter;
		
		// transpose the data matrix so that each row is a signal and the
		// columns contain instances (Fortran-ordered), then apply mean centering
		this.X = center(data.transpose());
		
		if (whiten) {
			this.X = whiten(X);
		}
		
		// get the size parameter of the symmetric W matrix
		num_components = Math.min(X.getColumnDimension(), X.getRowDimension());
		W = gaussianRandomSquareMatrix(num_components);
	}
	
	/**
	 * Default FastICA instance using the LogCosh() function to estimate
	 * negative entropy and whitening/mean-centering of the data matrix. This 
	 * implementation does not perform automatic component selection or reduction.
	 * 
	 * @param data - {@link Matrix} containing the source data; each column
	 * contains a single signal, and each row contains one sample of all signals 
	 * (rows contain instances, columns are features)
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 */
	public FastICA(Matrix data, float tolerance, int max_iter) {
		this(data, new LogCosh(), tolerance, max_iter, true);
	}
	
	
	public void train() {
		

		
	}
	
	/*
	 * FastICA main loop - default uses symmetric decorrelation. (i.e., 
	 * estimate all the independent components in parallel)
	 */
	private Matrix parallel_ica() {
		
		int rows = X.getRowDimension();
		int cols = X.getColumnDimension();
		Matrix W1 = new Matrix(rows, cols);
		double[][] W1_ = W1.getArray();
		
		W = symmetricOrthogonalization(W);
		for (int iter = 0; iter < max_iter; iter++) {
			G.apply(W.times(X));
			/*
			 * 1. A = gx.times(X.transpose())
			 * 2. A /= number of columns
			 * 3. A -= each g_x * the whole row of W
			 * 4. W1 = symmetricOrthogonalization(A)
			 * 5. lim = max(abs(abs(np.diag(dot(W1, W.T))) - 1)
			 */
			W1 = symmetricOrthogonalization();
			
			
		}
		return W;
	}
	
	/*
	 * Initialize the unmixing matrix from a standard Gaussian random generator
	 * with zero mean and unit variance. This implementation is ported from
	 * the sklearn python package.
	 */
	private Matrix gaussianRandomSquareMatrix(int size) {
		Matrix m = new Matrix(size, size);
		
		Random random = new Random();
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				m.set(i, j, random.nextGaussian());
			}
		}
		return m;
	}
	
	/*
	 * Center a matrix by subtracting the average of each row from every 
	 * element in the row
	 */
	private Matrix center(Matrix data) {
		double[][] x = data.getArrayCopy();
		int p = data.getColumnDimension();  // number of instances
		int n = data.getRowDimension();  // number of signals
		
		// calculate the mean of each row
		Matrix means = new Matrix(n, 1, 0);
		double[][] y = means.getArray();
		for (int i = 0; i < p; i++) {
			for (int j = 0; j < n; j++) {
				y[0][j] += x[i][j];
			}
		}
		means.timesEquals(1. / new Double(n));

		// subtract the means from each row
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				x[i][j] -= y[0][j];
			}
		}
		
		return new Matrix(x);
	}
	
	/*
	 * Return a whitened copy of a matrix by decorrelating and scaling the row 
	 * vectors. This implementation is ported from the sklearn python module 
	 * for fastica.
	 */
	private Matrix whiten(Matrix data) {
		Matrix x = data.copy();
		SingularValueDecomposition svd = x.svd();
		Matrix K = (svd.getU().arrayRightDivide(svd.getS())).transpose(); 
		x = K.times(x);
		x.timesEquals(1. / Math.sqrt(num_components));
		return x;
	}
	
	/*
	 * Return a copy of the input matrix of row vectors that has been updated 
	 * according to the rule for FastICA: B <- B.B' ^{-1/2} * B
	 * 
	 * there are only real eigenvalues because x.x' is Hermitian
	 */
	private Matrix symmetricOrthogonalization(Matrix x) {
		
		EigenvalueDecomposition ev = x.times(x.transpose()).eig();
		double[] e = ev.getRealEigenvalues();
		for (int i = 0; i < e.length; i++) {
			e[i] = Math.sqrt(e[i]);
		}
		
		// scale each column of the u matrix by the square root of the 
		// reciprocal of the associated eigenvalue to make the vectors 
		// unit length
		Matrix u = ev.getV().copy();
		double[][] u_ = u.getArray();
		for (int i = 0; i < u.getRowDimension(); i++) {
			for (int j = 0; j < u.getColumnDimension(); j++) {
				u_[i][j] /= e[j];
			}
		}
		return x.times(u.times(ev.getV().transpose()));
	}
	
}
