import java.io.IOException;
import java.util.Date;
import java.util.List;

import learn.nn.core.Example;
import learn.nn.core.NeuralNetwork;
import learn.nn.core.NeuralNetworkListener;
import learn.nn.examples.IrisExampleGenerator;
import learn.nn.examples.IrisNN;
import learn.nn.examples.MNIST;
import learn.nn.examples.MNISTNN;

public class MainIris{
	
	public static void main(String[] argv) throws IOException {
		int epochs = Integer.parseInt(argv[1]);//1000;
		double alpha = Double.parseDouble(argv[2]);//0.10;
		if(argv[0].equals("iris")){
			List<Example> examples = new IrisExampleGenerator("src/learn/nn/examples/iris.data.txt").examples();
			IrisNN network = new IrisNN();
			System.out.println("Training for " + epochs + " epochs with alpha=" + alpha);
			network.train(examples, epochs, alpha);
			network.dump();
			double accuracy = network.test(examples);
			System.out.println("Overall accuracy=" + accuracy);
			System.out.println();
			System.out.println("Confusion matrix:");
			double[][] matrix = network.confusionMatrix(examples);
			System.out.println("\tPredicted");
			System.out.print("Actual");
			for (int i=0; i < matrix.length; i++) {
				System.out.format("\t%d", i);
			}
			System.out.println();
			for (int i=0; i < matrix.length; i++) {
				System.out.format("%d", i);
				for (int j=0; j < matrix[i].length; j++) {
					System.out.format("\t%.3f", matrix[i][j]);
				}
				System.out.println();
			}
			System.out.println();
			int n = examples.size();
			int k = 10;
			System.out.println("k-Fold Cross-Validation: n=" + n + ", k=" + k);
			double acc = network.kFoldCrossValidate(examples, k, epochs, alpha);
			System.out.format("average accuracy: %.3f\n", acc);
			System.out.println();
			System.out.println("Learning Curve testing on all training data");
			System.out.println("EPOCHS\tACCURACY");
			for (epochs=100; epochs <= 3000; epochs+=100) {
				network.train(examples, epochs, alpha);
				accuracy = network.test(examples);
				System.out.format("%d\t%.3f\n",  epochs, accuracy);
			}
		}
		else if(argv[0].equals("mnist")){
			MNISTNN network = new MNISTNN();
			System.out.println("MNIST: reading training data...");
			String DATADIR = "src/learn/nn/examples";
			List<Example> trainingSet = MNIST.read(DATADIR+"/train-images-idx3-ubyte", DATADIR+"/train-labels-idx1-ubyte");
			System.out.println("MNIST: reading testing data...");
			List<Example> testingSet = MNIST.read(DATADIR+"/t10k-images-idx3-ubyte", DATADIR+"/t10k-labels-idx1-ubyte");
			System.out.println("MNIST: training on " + trainingSet.size() + " examples for " + epochs + " epochs, alpha=" + alpha);
			System.out.println("MNIST: testing on " + testingSet.size() + " examples");
			System.out.println("EPOCH\tACC\tTIMEms\tHHMMSS");;
			network.addListener(new NeuralNetworkListener() {
				protected long startTime;
				@Override
				public void trainingEpochStarted(NeuralNetwork network, int epoch) {
					startTime = new Date().getTime();
				}
				@Override
				public boolean trainingEpochCompleted(NeuralNetwork network, int epoch) {
					long now = new Date().getTime(); 
					double accuracy = network.test(testingSet);
					long elapsed = now - startTime;
					long s = elapsed / 1000;
					long h = s / (60*60);
					s -= h*60*60; 
					long m = s / 60;
					s -= m*60;
					System.out.format("%d\t%.3f\t%d\t%02d:%02d:%02d\n", epoch, accuracy, elapsed, h, m, s);
					return true;
				}
			});
			network.train(trainingSet, epochs, alpha);
		}
		
	}
}