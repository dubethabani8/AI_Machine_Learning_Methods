import java.io.IOException;
import java.util.List;

import learn.lc.core.DecayingLearningRateSchedule;
import learn.lc.core.Example;
import learn.lc.core.LogisticClassifier;
import learn.lc.core.PerceptronClassifier;
import learn.lc.examples.Data;

public class proj4Main {
	
	public static void main(String [] args) throws IOException {		

		int numIterations = Integer.parseInt(args[2]);
		List<Example> examples = Data.readFromFile("src/learn/lc/examples/"+args[1]);
		double learningRate = Double.parseDouble(args[3]);
		int numInputs = examples.get(0).inputs.length; 
		
		if(args[0].equals("logistic")) {
			LogisticClassifier classifier = new LogisticClassifier(numInputs) {
				@Override
				public void trainingReport(List<Example> examples, int stepnum, int nsteps) {
					double oneMinusError = 1.0-squaredErrorPerSample(examples);
					System.out.println(stepnum + "\t" + oneMinusError);
				}
			};
			if(learningRate == 0) classifier.train(examples, numIterations, new DecayingLearningRateSchedule());
			else classifier.train(examples, numIterations, learningRate);
		}
		
		else if(args[0].equals("perceptron")) { 
			PerceptronClassifier classifier = new PerceptronClassifier(numInputs) {
				@Override
				public void trainingReport(List<Example> examples, int stepnum, int nsteps) {
					double accuracy = accuracy(examples);
					System.out.println(stepnum + "\t" + accuracy);
				}
			};
			
			if(learningRate == 0) classifier.train(examples, numIterations, new DecayingLearningRateSchedule());
			else classifier.train(examples, numIterations, learningRate);
		}
	}
}