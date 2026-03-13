// code adapted from chatgpt
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
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
class XORNeuralNetwork implements ManipulateProvider {
  static final Distribution DISTRIBUTION = UniformDistribution.of(Clips.absolute(0.5));
  public static final Tensor X = Tensors.matrixInt(new int[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } }).unmodifiable();
  public static final Tensor XOR = Tensors.matrixInt(new int[][] { { 0 }, { 1 }, { 1 }, { 0 } }).unmodifiable();
  // ---
  @FieldSelectionArray({ "4", "5", "6", "7" })
  public Integer hiddenSize = 4;
  public Scalar l2 = RealScalar.of(0.00001);
  public Scalar learningRate = RealScalar.of(1.6);
  public Integer maxEpoch = 8000;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private static final int SKIP = 10;
    private final NetChain netChain = NetChains.binary(2, hiddenSize, 1);
    private final NetTrain netTrain;
    Scalar error;

    /** @param Y
     * @return error */
    Network(Tensor Y) {
      netChain.setL2(l2);
      netTrain = new NetTrain(netChain, X, Y);
      netTrain.run(learningRate, timeout, maxEpoch, SKIP);
      // ---
      error = FrobeniusNorm.of(netTrain.error());
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network(XOR);
    Scalar error = network.error;
    Show show1 = new Show();
    {
      TableBuilder table = network.netTrain.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(table.getColumns(0, i)));
      show1.setPlotLabel("Error: " + error.maps(Round._3));
    }
    Show show2 = new Show();
    {
      TableBuilder table = network.netTrain.tloss;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show2.add(ListLinePlot.of(table.getColumns(0, i)));
      show2.setPlotLabel("Error: " + error.maps(Round._3));
    }
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    new XORNeuralNetwork().runStandalone();
  }
}
