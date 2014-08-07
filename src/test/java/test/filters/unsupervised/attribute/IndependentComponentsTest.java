package test.filters.unsupervised.attribute;

import java.util.Random;


import org.ejml.simple.SimpleMatrix;
import org.junit.BeforeClass;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.CheckOptionHandler;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.IndependentComponents;

public class IndependentComponentsTest {
	
	static Instances signals;
	
	@BeforeClass
	public static void mixSignals() {
		// Read instances from data file containing 2000 samples mixing two
		// signals, sin(2t) + U[0,0.2] and square(3t) + U[0,0.2], with the
		// mixing matrix [[1,1], [0.5,2]]
		
		for (int j = 0; j < 3; j++) {
			signals.insertAttributeAt(new Attribute("S_" + Integer.toString(j)), j);
		}
		
	}
	
	@Test
	public void testFilter() {
		Instances result;
		IndependentComponents ICF = new IndependentComponents();
		
		try {
			result = Filter.useFilter(signals, ICF);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		// assert something about result
	}

}
