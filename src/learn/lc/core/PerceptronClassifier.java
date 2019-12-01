package learn.lc.core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import learn.math.util.VectorOps;

public class PerceptronClassifier extends LinearClassifier {
	protected PrintWriter out;

	public PerceptronClassifier(int ninputs) throws IOException {
		
		super(ninputs);
		out = new PrintWriter(new BufferedWriter(new FileWriter("output.txt")));

	}
	
	/**
	 * A PerceptronClassifier uses the perceptron learning rule
	 * (AIMA Eq. 18.7): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		
		// This must be implemented by you
		
		double h_w = threshold(VectorOps.dot(weights,x));  //dot product
		for(int i = 0; i < weights.length; i++)
			weights[i] = weights[i]+(alpha*(y - h_w)*x[i]);

	}
	
	/**
	 * A PerceptronClassifier uses a hard 0/1 threshold.
	 */
	public double threshold(double z) {
		
		
		// This must be implemented by you
		
		if(z<0) return 0;
		return 1;
	}
	
}
