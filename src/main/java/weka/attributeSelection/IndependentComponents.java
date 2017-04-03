package weka.attributeSelection;

import filters.FastICA;
import weka.core.*;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;

public class IndependentComponents extends UnsupervisedAttributeEvaluator implements AttributeTransformer, OptionHandler {
    protected FastICA m_filter;

    /** If true, whiten input data. */
    protected boolean m_whiten = true;

    /** Number of attributes to include. */
    protected int m_numAttributes = -1;

    /** Maximum number of FastICA iterations. */
    protected int m_numIterations = 200;

    /** Error tolerance for convergence. */
    protected double m_tolerance = 1E-4;

    @Override
    public void buildEvaluator(Instances instances) throws Exception {
        getCapabilities().testWithFail(instances);

        m_numAttributes = (m_numAttributes == -1) ? instances.numAttributes() : m_numAttributes;

        // TODO: remove class attribute if present
        // TODO: add filters to handle missing data and non-numeric attributes & remove the errors from determineOutputFormat

        // Convert the data to a row-indexed double[][]
        double[][] readings = new double[instances.numInstances()][];
        for (int i = 0; i < instances.numInstances(); i++) {
            readings[i] = instances.instance(i).toDoubleArray();
        }

        // Perform ICA
        m_filter = new FastICA(m_tolerance, m_numIterations, m_whiten);
        m_filter.fit(readings, instances.numAttributes());
    }

    @Override
    public double evaluateAttribute(int i) throws Exception {
        return 0;
    }

    @Override
    public Instances transformedHeader() throws Exception {
        return createTransformedInstances(0);
    }

    private Instances createTransformedInstances(int capacity) {
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        for (int i = 0; i < m_numAttributes; i++) {
            attributes.add(new Attribute("Source_" + Integer.toString(i)));
        }
        return new Instances("Source_ICA", attributes, capacity);
    }

    @Override
    public Instances transformedData(Instances instances) throws Exception {
        Instances output = createTransformedInstances(instances.numInstances());
        for (int i=0; i < instances.numInstances(); i++) {
            output.add(convertInstance(instances.get(i)));
        }
        return output;
    }

    @Override
    public Instance convertInstance(Instance currentInstance) throws Exception {
        int last_idx;
        Instance inst;
        double[][] result;

        if (m_filter == null) {
            throw new Exception("No ICA instance has been trained.");
        }

        result = m_filter.transform(new double[][]{currentInstance.toDoubleArray()});
        last_idx = result[0].length;

        // copy the results back into an instance
        inst = new DenseInstance(m_numAttributes);
        for (int i = 0; i < last_idx; i++) {
            inst.setValue(i, result[0][i]);
        }

        return inst;
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

    public String numAttributesTipText() {
        return "Number of separate sources to identify in the output." +
                " (-1 = include all; default: -1)";
    }

    public void setNumAttributes(int num) {
        m_numAttributes = num;
    }

    public int getNumAttributes() {
        return m_numAttributes;
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
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);

        return result;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return      an enumeration of all the available options.
     */
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
    public void setOptions(String[] options) throws Exception {
        String        tmpStr;

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

        tmpStr = Utils.getOption('T', options);
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

        result.add("-W");

        result.add("-A");
        result.add("" + getNumAttributes());

        result.add("-N");
        result.add("" + getNumIterations());

        result.add("-T");
        result.add("" + getTolerance());

        return result.toArray(new String[result.size()]);
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 1.0 $");
    }

    /**
     * Returns a description of this attribute transformer
     *
     * @return a String describing this attribute transformer
     */
    @Override
    public String toString() {
        if (m_filter == null) {
            return "No ICA instance has been trained.";
        } else {
            return "\tIndependent Components Attribute Transformer\n\n" + independentComponentsSummary();
        }
    }

    private String independentComponentsSummary() {
        StringBuffer results = new StringBuffer();
        results.append("Kurtosis\n\n");
        return results.toString();
    }

    /**
     * Main method for executing this evaluator.
     *
     * @param args the options, use "-h" to display options
     */
    public static void main(String[] args) {
        ASEvaluation.runEvaluator(new IndependentComponents(), args);
    }
}
