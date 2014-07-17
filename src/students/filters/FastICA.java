package students.filters;

import java.util.Arrays;
import java.util.Random;

import org.ejml.data.MatrixIterator;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;



/**
 * FastICA port from scikit-learn implementation.
 * 
 * @author Chris Gearhart <cgearhart3@gatech.edu>
 *
 */
public class FastICA {
	
	// 
	private SimpleMatrix S;
	
	// The data matrix
	private SimpleMatrix X;
	
	// The unmixing matrix
	private SimpleMatrix W;
	
	// Reference to non-linear neg-entropy estimator function
	private ICAGFunction G;
	
	// Convergence tolerance
	private final double tolerance;
	
	// Iteration limit
	private final int max_iter;
	
	// number of components to output
	private final int num_components;
	
	/**
	 * General FastICA instance constructor with an arbitrary (user-supplied)
	 * function to estimate negative entropy. This implementation does not
	 * perform automatic component selection or reduction.
	 * 
	 * @param data - {@link SimpleMatrix} containing the source data; each column
	 * contains a single signal, and each row contains one sample of all signals 
	 * (rows contain instances, columns are features)
	 * @param g_func - {@link ICAGFunction} to estimate negative entropy 
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 * @param whiten - whiten the data matrix (default true)
	 */
	public FastICA(SimpleMatrix data, ICAGFunction g_func, double tolerance, int max_iter, boolean whiten) {
		this.G = g_func;
		this.tolerance = tolerance;
		this.max_iter = max_iter;
		
		this.X = center(data);
		
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
	 * @param data - {@link SimpleMatrix} containing the source data; each column
	 * contains a single signal, and each row contains one sample of all signals 
	 * (rows contain instances, columns are features)
	 * @param tolerance - maximum allowable convergence error
	 * @param max_iter - max number of iterations
	 */
	public FastICA(SimpleMatrix data, double tolerance, int max_iter) {
		this(data, new LogCosh(), tolerance, max_iter, true);
	}
	
//	public Matrix transform() {
//		
//	}
	
	/*
	 * FastICA main loop - default uses symmetric decorrelation. (i.e., 
	 * estimate all the independent components in parallel)
	 */
	public Matrix parallel_ica() {
		
		int m = X.getRowDimension();
		int n = X.getColumnDimension();
		Matrix W1 = new Matrix(m, n);
		double[][] W1_ = W1.getArray();
		
		W = symmetricOrthogonalization(W);
		for (int iter = 0; iter < max_iter; iter++) {
			G.apply(W.times(X));
			Matrix gx = G.getGx();
			Matrix A = gx.times(X.transpose());
			A.timesEquals(1. / new Double(n));
			A.minusEquals(rowMultiply(W, G.getGpx()));
			W1 = symmetricOrthogonalization(A);
			
			double lim = max(abs(add(abs(diag(W1.times(W.transpose()))),-1)));
			W = W1;
			if (lim < tolerance) 
				break;
		}
		return W;
	}
	
//	private Matrix rowMultiply(Matrix A, Matrix B) {
//		double[][] _A = A.getArrayCopy();
//		double[][] _B = B.getArray();
//		
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				_A[i][j] *= _B[i][0];
//			}
//		}
//		return new Matrix(_A);
//	}
//	
//	private Matrix colMultiply(Matrix A, Matrix B) {
//		double[][] _A = A.getArrayCopy();
//		double[][] _B = B.getArray();
//
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				_A[i][j] *= _B[0][j];
//			}
//		}
//		return new Matrix(_A);
//	}
//	
//	private Matrix recip(Matrix A) {
//		double[][] _A = A.getArrayCopy();
//		
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				_A[i][j] = 1. / _A[i][j];
//			}
//		}
//		return new Matrix(_A);
//	}
//	
//	private Matrix diag(Matrix A) {
//		int l = Math.min(A.getColumnDimension(), A.getRowDimension());
//		Matrix x = new Matrix(1, l, 0);
//		double[][] x_ = x.getArray();
//		for (int i = 0; i < l; i++) {
//			x_[0][i] = A.get(i, i);
//		}
//		return x;
//	}
//	
//	private double max(Matrix A) {
//		double[][] _A = A.getArray();
//		double x = 0;
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				if (_A[i][j] > x) {
//					x = _A[i][j];
//				}
//			}
//		}
//		return x;
//	}
//	
//	private Matrix abs(Matrix A) {
//		double[][] _A = A.getArrayCopy();
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				_A[i][j] = Math.abs(_A[i][j]);
//			}
//		}
//		return new Matrix(_A);
//	}
//	
//	private Matrix add(Matrix A, double x) {
//		double[][] _A = A.getArrayCopy();
//		for (int i = 0; i < A.getRowDimension(); i++) {
//			for (int j = 0; j < A.getColumnDimension(); j++) {
//				_A[i][j] -= x;
//			}
//		}
//		return new Matrix(_A);
//	}
	
	/*
	 * Generate a matrix filled with standard Gaussian random variables
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
	 * Center a matrix by subtracting the average of each column vector
	 * from every element in the column
	 */
	public static SimpleMatrix center(SimpleMatrix x) {
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
			x.insertIntoThis(0, i, col);
		}
		return x;
	}
	
	/*
	 * Whiten a matrix, x,  of column vectors by decorrelating and 
	 * scaling the elements according to: x_ = ED^{-1/2}E'x , where E is the
	 * orthogonal matrix of eigenvectors of E{xx'}. In this implementation,
	 * the eigen decomposition is replaced with the SVD based on the FastICA
	 * implementation in the sklearn Python package.
	 * 
	 * Testing this method is difficult because the decomposition is ambiguous
	 * with regard to the direction of column vectors - they can be +/-.
	 */
	public static SimpleMatrix whiten(SimpleMatrix x) {
		// get compact SVD (D matrix is min(m,n) square)
		SimpleSVD svd = x.svd(true);   
		
		// TODO: K should only keep `num_components` columns if performing
		// dimensionality reduction
		SimpleMatrix K = svd.getV().mult(svd.getW().invert());
	
		SimpleMatrix x_ = x.mult(K);
		x_ = x_.scale(Math.sqrt(x.numRows()));
		return x_;
	}
	
	/*
	 * Return a copy of the input matrix of row vectors that has been updated 
	 * according to the rule for FastICA: B <- B.B' ^{-1/2} * B, from the
	 * sklearn python module for fastica.
	 * 
	 * NOTE: There are only real eigenvalues because x.x' is Hermitian
	 */
	public static SimpleMatrix symmetricDecorrelation(SimpleMatrix x) {
		
		
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
