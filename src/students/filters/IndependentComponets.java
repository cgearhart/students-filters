package students.filters;

import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.filters.SimpleBatchFilter;

public class IndependentComponets extends SimpleBatchFilter {
	
	static final long serialVersionUID = 0;

	@Override
	public String globalInfo() {
		return "Performs Independent Component Analysis and transformation " +
				"of the data using the FastICA algorithm ignoring the class " +
				"label.";
	}

	@Override
	protected Instances determineOutputFormat(Instances inputFormat)
			throws Exception {
		return new Instances(inputFormat, 0);
	}

	@Override
	protected Instances process(Instances instances) throws Exception {
		// needs args for max_iter, tolerance
		
		// Process instances to matrix ignoring class labels
		

		// Process matrix back to Instances
		
		return null;
	}
	
	public String getRevision() {
	    return RevisionUtils.extract("$Revision: 1.0 $");
	  }

}
