package students.filters;

import weka.core.Attribute;
import java.util.Enumeration;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionUtils;
import weka.core.Capabilities.Capability;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;

public class IndependentComponetFilter 
	extends SimpleBatchFilter
	implements OptionHandler, UnsupervisedFilter{
	/** for serialization. */
	private static final long serialVersionUID = -5416810876710954131L;
	
	/** If true, center input data. */
	protected boolean m_center = true;
	
	/** If true, whiten input data. */
	protected boolean m_whiten = true;
	
	/** Number of attributes to include. */
	protected int m_numAttributes = -1;
	
	/** Maximum number of FastICA iterations. */
	protected int m_numIterations = 200;
	
	/** Error tolerance for convergence. */
	protected double m_tolerance = 1E-4;

	@Override
	public String globalInfo() {
		return "Performs Independent Component Analysis and transformation " +
				"of the data using the FastICA algorithm ignoring the class " +
				"label.";
	
	public void setCenterData(boolean flag) {
		m_center = flag;
	}
	
	public boolean getCenterData() {
		return m_center;
	}
	
	public void setWhitenData(boolean flag) {
		m_whiten = flag;
	}
	
	public boolean getWhitenData() {
		return m_whiten;
	}
	
	public void setNumAttributes(int num) {
		m_numAttributes = num;
	}
	
	public int getNumAttributes() {
		return m_numAttributes;
	}
	
	public void setNumIterations(int num) {
		m_numIterations = num;
	}
	
	public int getNumIterations() {
		return m_numIterations;
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
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
	
		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		
		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
		result.enable(Capability.NO_CLASS);
		
		return result;
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		return new Instances(inputFormat, 0);
	}
	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return      an enumeration of all the available options.
	 */
	public Enumeration listOptions() {
		Vector<Option> result = new Vector<Option>();
	 
		result.addElement(new Option(
				"\tCenter the data (subtract the mean of each attribute from\n" +
				"\teach instance) before performing ICA. This should *probably*\n" +
				"\tbe enabled unless you already centered the data.\n",
				"C", 0, "-C"));

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
	 * <pre> -C
	 *  Center the data (subtract the mean of each attribute from
	 *  each instance) before performing ICA. This should *probably*
	 *  be enabled unless you already centered the data.</pre>
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
	public void setOptions(String[] options) throws Exception {
		String        tmpStr;
		
		setCenterData(Utils.getFlag('C', options));
		
		setWhitenData(Utils.getFlag('W', options));

		tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0)
			setNumAttributes(Integer.parseInt(tmpStr));
		else
			setNumAttributes(-1);

		tmpStr = Utils.getOption('N', options);
		if (tmpStr.length() != 0)
			setNumIterations(Integer.parseInt(tmpStr));
		else
			setNumIterations(200);

		tmpStr = Utils.getOption('N', options);
		if (tmpStr.length() != 0)
			setTolerance(Double.parseDouble(tmpStr));
		else
			setTolerance(1E-4);   
	}

	/**
	 * Gets the current settings of the filter.
	 *
	 * @return      an array of strings suitable for passing to setOptions
	 */
	public String[] getOptions() {
		Vector<String>   result;

		result = new Vector<String>();

		result.add("-C");
		result.add("" + getCenterData());
		
		result.add("-W");
		result.add("" + getWhitenData());

		result.add("-A");
		result.add("" + getNumAttributes());
		
		result.add("-N");
		result.add("" + getNumIterations());
		
		result.add("-T");
		result.add("" + getTolerance());

		return result.toArray(new String[result.size()]);
	}
	@Override
	protected Instances process(Instances instances) throws Exception {
		
		// Test that there is no missing data and only contains numeric attributes
				
		// TODO: needs args for max_iter, tolerance
		
		// Convert instances to matrix ignoring class labels
		
		// Process the data through FastICA

		// Convert results back to Instances
		
		return null;
	}
	
	public String getRevision() {
	    return RevisionUtils.extract("$Revision: 1.0 $");
	  }

}
