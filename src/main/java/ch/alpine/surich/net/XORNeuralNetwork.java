// code adapted from chatgpt
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldSelectionArray;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.nrm.FrobeniusNorm;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Round;

/** Quote from chatgpt:
 * 
 * This example:
 * Uses 1 hidden layer
 * Uses sigmoid activation
 * Trains with stochastic gradient descent
 * Prints evaluation after training
 * 
 * The example network has 3 layers total:
 * Input layer – 2 neurons
 * Inputs: x1, x2
 * Hidden layer – 4 neurons
 * Uses sigmoid activation
 * This is what allows the network to solve XOR (non-linear problem)
 * Output layer – 1 neuron
 * Produces the XOR result
 * Also uses sigmoid activation */
@ReflectionMarker
public class XORNeuralNetwork implements ManipulateProvider {
  static final Distribution DISTRIBUTION = UniformDistribution.of(Clips.absolute(0.5));
  public static final Tensor X = Tensors.matrixInt(new int[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }).unmodifiable();
  public static final Tensor XOR = Tensors.matrixInt(new int[][] { { 0 }, { 1 }, { 1 }, { 0 } }).unmodifiable();
  public static final int SKIP = 10;
  // ---
  @FieldSelectionArray({ "4", "5", "6", "7" })
  public Integer hiddenSize = 4;
  public Scalar l2 = RealScalar.of(0.00001);
  public Scalar learningRate = RealScalar.of(1.6);
  public Integer maxEpoch = 8000;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private final NetChain netChain = NetChains.binary(2, hiddenSize, 1);
    private final TableBuilder tableBuilder = new TableBuilder();

    void train(Tensor y) {
      netChain.setL2(l2);
      NetTrain.of(netChain, X, y, learningRate, tableBuilder::appendRow, timeout, maxEpoch, SKIP);
    }

    Scalar evaluate(Tensor Y) {
      System.out.println("Evaluation after training:");
      Tensor errors = Tensors.empty();
      for (int sample = 0; sample < X.length(); ++sample) {
        Tensor x = X.get(sample);
        Tensor y = netChain.forward(x);
        Tensor t = Y.get(sample);
        Tensor e = t.subtract(y);
        errors.append(e);
        System.out.printf("Input: %s -> Error: %s\n", x, e.maps(Round._3));
      }
      return FrobeniusNorm.of(errors);
    }
  }

  public Show getShow() {
    Network network = new Network();
    network.train(XOR);
    Scalar error = network.evaluate(XOR);
    IO.println(error);
    Tensor table = network.tableBuilder.getTable();
    IO.println(Dimensions.of(table));
    int n = Unprotect.dimension1Hint(table);
    Tensor domain = Range.of(0, table.length()).multiply(RealScalar.of(SKIP));
    Show show = new Show();
    for (int i = 0; i < n; ++i)
      show.add(ListLinePlot.of(domain, table.get(Tensor.ALL, i)));
    return show;
  }

  @Override
  public Container getContainer() {
    return ShowGridComponent.of(getShow());
  }

  static void main() {
    new XORNeuralNetwork().runStandalone();
  }
}
