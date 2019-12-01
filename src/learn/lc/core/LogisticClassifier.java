package learn.lc.core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import learn.math.util.VectorOps;

public class LogisticClassifier extends LinearClassifier {
	protected PrintWriter out;
	public LogisticClassifier(int ninputs) throws IOException {
		super(ninputs);
		out = new PrintWriter(new BufferedWriter(new FileWriter("output.txt")));
	}
	
	/**
	 * A LogisticClassifier uses the logistic update rule
	 * (AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i 
	 */
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		
		double h_w = threshold(VectorOps.dot(weights,x));
		for(int i = 0; i < weights.length; i++)
			weights[i] = weights[i]+ (alpha*(y-h_w)*(h_w)*(1-h_w)*x[i]);
	}
	
	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	public double threshold(double z) {
		// This must be implemented by you
		return 1/(1+Math.pow(Math.E, -z));

	}

}
