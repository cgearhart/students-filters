package test.filters.unsupervised.attribute;

import static org.junit.Assert.assertEquals;

import java.io.BufferedReader;
import java.io.FileReader;

import org.junit.BeforeClass;
import org.junit.Test;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.filters.unsupervised.attribute.IndependentComponents;

public class IndependentComponentsTest {
	
	static Instances m_data;
	static int m_numAttrs; 
	
	@BeforeClass
	public static void readArff() throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader("iris.arff"));
		ArffReader arff = new ArffReader(reader);
		m_data = arff.getData();
		m_data.setClassIndex(m_data.numAttributes() - 1);
		m_numAttrs = 2;  // valid choices are 1-4 for iris data set
	}
	
	@Test
	public void testFilter() throws Exception {
		IndependentComponents filter = new IndependentComponents();
		filter.setInputFormat(m_data);
		filter.setOutputNumAtts(m_numAttrs); // optionally set the number of attributes (for dimensionality reduction)
		for (int i = 0; i < m_data.numInstances(); i++) {
			filter.input(m_data.instance(i));
		}
		filter.batchFinished();
		Instances newData = filter.getOutputFormat();
		Instance processed;
		while ((processed = filter.output()) != null) {
			newData.add(processed);
		}

		// m_numAttrs = numAttributes() - 1 because numAttributes() includes the class attribute
		assertEquals(m_numAttrs, newData.numAttributes() - 1);
		assertEquals(m_data.numClasses(), newData.numClasses());
		assertEquals(m_data.numInstances(), newData.numInstances());
	}

}
