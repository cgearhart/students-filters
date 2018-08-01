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

package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

import filters.FastICA;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;

/**
 * <!-- globalinfo-start --> Performs independent components analysis and
 * transformation of the data.<br/>
 * ICA does not perform dimensionality reduction by default; run PCA (or similar)
 * to determine the number of components to keep, then run ICA to find that 
 * number of attributes.<br/>
 * Based on code of the filter 'PrincipalComponents' fracpete
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -W
 *  Whiten the data (decorrelate the inputs so
 *  the covariance is the identity matrix).
 * </pre>
 * 
 * <pre>
 * -A &lt;num&gt;
 *  Maximum number of attributes to include in results.
 *  (-1 = include all, default: -1)
 * </pre>
 * 
 * <pre>
 * -N &lt;num&gt;
 *  Maximum number of iterations.
 *  (default: 200)
 * </pre>
 * 
 * <pre>
 * -T &lt;num&gt;
 *  Convergence tolerance to stop training.
 *  (default: 1E-4)
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Chris Gearhart (cgearhart3@gatech.edu)
 * @version $Revision: 12660 $
 */ 
public class IndependentComponents 
	extends Filter
	implements OptionHandler, UnsupervisedFilter{
	
	/** for serialization. */
	private static final long serialVersionUID = -5416810876710954131L;
	
	protected FastICA m_filter;
	
	/** The header for the transformed data format. */
	protected Instances m_TransformedFormat;
	
	/** The data to transform analyse/transform. */
	protected Instances m_TrainInstances;
	
	/** Keep a copy for the class attribute (if set). */
	protected Instances m_TrainCopy;
	
	/** True when the instances sent to determineOutputFormat() has a class attribute */
	protected boolean m_HasClass;
	
	/** Class index. */
	protected int m_ClassIndex;
	
	/** Number of attributes in input. */
	protected int m_NumAttribs;
	
	/** Number of instances in input. */
	protected int m_NumInstances;
	
	/** If true, whiten input data. */
	protected boolean m_whiten = true;
	
	/** Maximum number of FastICA iterations. */
	protected int m_numIterations = 200;
	
	/** Error tolerance for convergence. */
	protected double m_tolerance = 1E-4;
	
	/** Filters for replacing missing values. */
	protected ReplaceMissingValues m_ReplaceMissingFilter;

	/** Filter for turning nominal values into numeric ones. */
	protected NominalToBinary m_NominalToBinaryFilter;

	/** Filter for removing class attribute, nominal attributes with 0 or 1 value. */
	protected Remove m_AttributeFilter;
	
	/** The number of attributes in the transformed data (-1 for all). */
	protected int m_OutputNumAtts = -1;

	public String globalInfo() {
		return "Performs Independent Component Analysis and transformation " +
				"on numeric data using the FastICA algorithm. The class label " +
				"is ignored.";
	}
	
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return      an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();

		result.addElement(new Option(
			"\tWhiten the data (decorrelate the inputs so the covariance\n" +
			"\tis the identity matrix) before performing ICA. This should\n" +
			"\t*probably* be enabled unless you whitened the data yourself\n" +
			"\tor have a specific reason not to perform whitening.",
			"W", 0, "-W"));

		result.addElement(new Option(
			"\tMaximum number of attributes to include in results.\n" +
			"\t(-1 = include all, default: all)", 
			"A", 1, "-A <num>"));

		result.addElement(new Option(
			"\tMaximum number of iterations.\n\t(default: 200)", 
			"N", 1, "-N <num>"));
		
		result.addElement(new Option(
			"\tConvergence tolerance to stop training.\n\t(default: 1E-4)", 
			"T", 1, "-T <num>"));

		return result.elements();
	}
	
	/**
	 * Parses a list of options for this object. <p/>
	 *
	 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -W
	 *  Whiten the data (decorrelate the inputs so the covariance
	 *  is the identity matrix) before performing ICA. This should
	 *  *probably* be enabled unless you whitened the data yourself
	 *  or have a specific reason not to perform whitening.</pre>
	 * 
	 * <pre> -A &lt;num&gt;
	 *  Maximum number of attributes to include in results.
	 *  (-1 = include all, default: all)</pre>
	 *  
	 * <pre> -N &lt;num&gt;
	 *  Maximum number of iterations.
	 *  (default: 200)</pre>
	 *  
	 * <pre> -T &lt;num&gt;
	 *  Convergence tolerance to stop training.
	 *  (default: 1E-4)</pre>
	 * 
	 <!-- options-end -->
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		String        tmpStr;

		setWhitenData(Utils.getFlag('W', options));

		tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0)
			setOutputNumAtts(Integer.parseInt(tmpStr));
		else
			setOutputNumAtts(m_OutputNumAtts);

		tmpStr = Utils.getOption('N', options);
		if (tmpStr.length() != 0)
			setNumIterations(Integer.parseInt(tmpStr));
		else
			setNumIterations(m_numIterations);

		tmpStr = Utils.getOption('T', options);
		if (tmpStr.length() != 0)
			setTolerance(Double.parseDouble(tmpStr));
		else
			setTolerance(m_tolerance);
	}

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return      an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {
		Vector<String>   result;

		result = new Vector<String>();

		result.add("-W");

		result.add("-A");
		result.add("" + getOutputNumAtts());
		
		result.add("-N");
		result.add("" + getNumIterations());
		
		result.add("-T");
		result.add("" + getTolerance());

		return result.toArray(new String[result.size()]);
	}
	
	public String whitenDataTipText() {
		return "Whiten the data (decoupling transform) if set.";
	}
	
	public void setWhitenData(boolean flag) {
		m_whiten = flag;
	}
	
	public boolean getWhitenData() {
		return m_whiten;
	}
	
	public String outputNumAttsTipText() {
		return "Number of separate sources to identify in the output. (-1 = include all; default: -1)";
	}
	
	public void setOutputNumAtts(int num) {
		m_OutputNumAtts = num;
	}
	
	public int getOutputNumAtts() {
		return m_OutputNumAtts;
	}
	
	public String numIterationsTipText() {
		return "The maximum number of iterations of the FastICA main loop to allow.";
	}
	
	public void setNumIterations(int num) {
		m_numIterations = num;
	}
	
	public int getNumIterations() {
		return m_numIterations;
	}
	
	public String toleranceTipText() {
		return "Error tolerance for solution convergence.";
	}
	
	public void setTolerance(double tolerance) {
		m_tolerance = tolerance;
	}
	
	public double getTolerance() {
		return m_tolerance;
	}
	
	/**
	 * Returns the capabilities of this evaluator.
	 *
	 * @return            the capabilities of this evaluator
	 * @see               Capabilities
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
	
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	    result.enable(Capability.DATE_ATTRIBUTES);
	    result.enable(Capability.MISSING_VALUES);
		
		// class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.UNARY_CLASS);
	    result.enable(Capability.NUMERIC_CLASS);
	    result.enable(Capability.DATE_CLASS);
	    result.enable(Capability.MISSING_CLASS_VALUES);
	    result.enable(Capability.NO_CLASS);
		
		return result;
	}

	/**
	 * Determines the output format based on the input format and returns 
	 * this. In case the output format cannot be returned immediately, i.e.,
	 * immediateOutputFormat() returns false, then this method will be called
	 * from batchFinished().
	 *
	 * @param inputFormat     the input format to base the output format on
	 * @return                the output format
	 * @throws Exception      in case the determination goes wrong
	 * @see   #batchFinished()
	 */
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		
		if (m_OutputNumAtts < 0 || m_OutputNumAtts > inputFormat.numAttributes()) {
			m_OutputNumAtts = inputFormat.numAttributes();
		}
		
		// Add generic labels for the required number of unmixed sources
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		for (int i = 0; i < m_OutputNumAtts; i++) { 
			attributes.add(new Attribute("Source_" + Integer.toString(i)));
		}
		
		if (m_HasClass) {
			attributes.add((Attribute) m_TrainCopy.classAttribute().copy());
		}
	    
	    Instances outputFormat = new Instances(inputFormat.relationName()
	    		+ "_ICA", attributes, 0);
	    
	    // set the class to be the last attribute if necessary
	    if (m_HasClass) {
	    	outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
	    }
	    
	    return outputFormat;
	}
	
	/**
	 * 
	 * Transform a single instance into the ICA projected space based on the
	 * FastICA filter object. This should not be called prior to training the
	 * filter with input(), then calling batchFinished().
	 * 
	 * @param currentInstance 		an {@link Instance} object containing one value of 
	 * 				each attribute
	 * @return 				a {@link Instance} object containing the result of 
	 * 				projecting the input into the ICA domain
	 * @throws Exception	if the filter has not been trained, or the
	 * 				transformation does not succeed
	 */
	protected Instance convertInstance(Instance currentInstance) throws Exception {
		Instance result;
		double[] tmpVals;
		double[] newVals;
		Instance tempInst;

		// make a copy of the instance and transform it with the trained ICA filter
		tempInst = (Instance) currentInstance.copy();

		m_ReplaceMissingFilter.input(tempInst);
	    m_ReplaceMissingFilter.batchFinished();
	    tempInst = m_ReplaceMissingFilter.output();

	    m_NominalToBinaryFilter.input(tempInst);
	    m_NominalToBinaryFilter.batchFinished();
	    tempInst = m_NominalToBinaryFilter.output();

	    if (m_AttributeFilter != null) {
	      m_AttributeFilter.input(tempInst);
	      m_AttributeFilter.batchFinished();
	      tempInst = m_AttributeFilter.output();
	    }

	    double[][] data = new double[][]{tempInst.toDoubleArray()};
		tmpVals = m_filter.transform(data)[0];
		
		if (m_HasClass) {
			newVals = new double[m_OutputNumAtts + 1];
		} else {
			newVals = new double[m_OutputNumAtts];
		}
		System.arraycopy(tmpVals, 0, newVals, 0, tmpVals.length);
		
		if (m_HasClass) {
			newVals[m_OutputNumAtts] = currentInstance.value(currentInstance.classIndex());
		}

	    if (currentInstance instanceof SparseInstance) {
	    	result = new SparseInstance(currentInstance.weight(), newVals);
	    } else {
	    	result = new DenseInstance(currentInstance.weight(), newVals);
	    }
		
		return result;
	}
	
	/**
	 * Initialize the filter on the input data
	 * 
	 * @param	instances 		{@link Instances} containing data for
	 * 					training/fitting the {@link FastICA} filter
	 * @throws	Exception 		if the {@link FastICA} or any preprocessing 
	 * 					filters encounter a problem
	 */
	protected void setup(Instances instances) throws Exception {
		int i;
		Vector<Integer> deleteCols;
	    int[] todelete;

		m_TrainInstances = new Instances(instances);
		
		// make a copy of the training data so that we can get the class
	    // column to append to the transformed data (if necessary)
	    m_TrainCopy = new Instances(m_TrainInstances, 0);
		
	    m_ReplaceMissingFilter = new ReplaceMissingValues();
	    m_ReplaceMissingFilter.setInputFormat(m_TrainInstances);
	    m_TrainInstances = Filter.useFilter(m_TrainInstances,
	      m_ReplaceMissingFilter);

	    m_NominalToBinaryFilter = new NominalToBinary();
	    m_NominalToBinaryFilter.setInputFormat(m_TrainInstances);
	    m_TrainInstances = Filter.useFilter(m_TrainInstances,
	      m_NominalToBinaryFilter);

	    // delete any attributes with only one distinct value or are all missing
	    deleteCols = new Vector<Integer>();
	    for (i = 0; i < m_TrainInstances.numAttributes(); i++) {
	      if (m_TrainInstances.numDistinctValues(i) <= 1) {
	        deleteCols.addElement(i);
	      }
	    }

	    if (m_TrainInstances.classIndex() >= 0) {
	      // get rid of the class column
	      m_HasClass = true;
	      m_ClassIndex = m_TrainInstances.classIndex();
	      deleteCols.addElement(new Integer(m_ClassIndex));
	    }

	    // remove columns from the data if necessary
	    if (deleteCols.size() > 0) {
	      m_AttributeFilter = new Remove();
	      todelete = new int[deleteCols.size()];
	      for (i = 0; i < deleteCols.size(); i++) {
	        todelete[i] = (deleteCols.elementAt(i)).intValue();
	      }
	      m_AttributeFilter.setAttributeIndicesArray(todelete);
	      m_AttributeFilter.setInvertSelection(false);
	      m_AttributeFilter.setInputFormat(m_TrainInstances);
	      m_TrainInstances = Filter.useFilter(m_TrainInstances, m_AttributeFilter);
	    }

	    // can evaluator handle the processed data ? e.g., enough attributes?
	    getCapabilities().testWithFail(m_TrainInstances);

	    m_NumInstances = m_TrainInstances.numInstances();
	    m_NumAttribs = m_TrainInstances.numAttributes();
		
		// Convert the data to a row-indexed double[][]
		double[][] readings = new double[m_NumInstances][];
		for (i = 0; i < m_NumInstances; i++) {
			readings[i] = m_TrainInstances.instance(i).toDoubleArray();
		}
		
		m_TransformedFormat = determineOutputFormat(m_TrainInstances);
	    setOutputFormat(m_TransformedFormat);

		// Perform ICA
		m_filter = new FastICA(m_tolerance, m_numIterations, m_whiten);
		m_filter.fit(readings, m_OutputNumAtts);

	    m_TrainInstances = null;
	}

	/**
	 * Sets the format of the input instances.
	 *
	 * @param instanceInfo    an {@link Instances} object containing the input 
	 *                instance structure (any instances contained 
	 *                in the object are ignored - only the structure 
	 *                is required).
	 * @return            true if the outputFormat may be collected 
	 *                immediately
	 * @throws Exception      if the input format can't be set successfully
	 */
	@Override
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
		super.setInputFormat(instanceInfo);

		m_filter = null;
		m_HasClass = false;
	    m_AttributeFilter = null;
	    m_NominalToBinaryFilter = null;
		
		return false;
	}

	/**
	 * Input an instance for filtering. Filter requires all training instances be
	 * read before producing output.
	 * 
	 * @param instance the input instance
	 * @return true if the filtered instance may now be collected with output().
	 * @throws IllegalStateException if no input format has been set
	 * @throws Exception if conversion fails
	 */
	@Override
	public boolean input(Instance instance) throws Exception {
		Instance inst;

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		if (isNewBatch()) {
			resetQueue();
			m_NewBatch = false;
		}

		if (isFirstBatchDone()) {
			inst = convertInstance(instance);
			inst.setDataset(getOutputFormat());
			push(inst, false);
			return true;
		} else {
			bufferInput(instance);
			return false;
		}
	}
	
	/**
	 * Signify that this batch of input to the filter is finished. If
	 * the filter requires all instances prior to filtering, output()
	 * may now be called to retrieve the filtered instances. Any
	 * subsequent instances filtered should be filtered based on setting
	 * obtained from the first batch (unless the inputFormat has been
	 * re-assigned or new options have been set). This default
	 * implementation assumes all instance processing occurs during
	 * inputFormat() and input().
	 *
	 * @return 	true 					if there are instances pending output
	 * @throws 	NullPointerException 	if no input structure has 
	 * 						been defined
	 * @throws 	Exception 				if there was a problem finishing the batch
	 */
	@Override
	public boolean batchFinished() throws Exception {
		Instances inputs;
		Instance output;
		
		if (getInputFormat() == null) {
			throw new Exception("Input format not defined.");
		}
		
		inputs = getInputFormat();
		
		if (!isFirstBatchDone()) {
			setup(inputs);
		}
		
		for (int i = 0; i < inputs.numInstances(); i++) {
			output = convertInstance(inputs.instance(i));
			output.setDataset(getOutputFormat());
			push(output, false);
		}
		
		flushInput();
	    m_NewBatch = true;
	    m_FirstBatchDone = true;
		
	    return (numPendingOutput() != 0);
	}
	
	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 2.0 $");
	}
	
	/**
	 * Main method for running this filter.
	 *
	 * @param args 	should contain arguments to the filter: use -h for help
	 */
	public static void main(String[] args) {
		runFilter(new IndependentComponents(), args);
	}

}
