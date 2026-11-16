// code adapted from chatgpt
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldClip;
import ch.alpine.bridge.ref.ann.FieldSelectionArray;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.ElementwiseLayer;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetMomentumGradientDescent;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.TableBuilder;
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
  public static final TrainingData TRAINING_DATA = new TrainingData(X, XOR);
  // ---
  @FieldSelectionArray({ "4", "5", "6", "7" })
  public Integer hiddenSize = 4;
  public Scalar l2 = RealScalar.of(0.00001);
  public Scalar beta = RealScalar.of(0.9);
  public Scalar eta = RealScalar.of(0.02);
  @FieldClip(min = "1", max = "4")
  public Integer batch_size = 1;
  public Integer maxEpoch = 2000;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    final NetLayer netLayer = //
        NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, hiddenSize, 1));
    final NetTrain netTrain;
    final NetParamTracker netParamTracker = new NetParamTracker();

    /** @param Y */
    Network(Tensor Y) {
      NetInitialize.of(netLayer);
      NetOptimizer netOptimize = new NetMomentumGradientDescent(netLayer, beta, eta);
      netTrain = new NetTrain(netLayer, netOptimize);
      netTrain.addListener(netParamTracker);
      netTrain.setMaxEpochs(maxEpoch);
      netTrain.runAsBatches(TRAINING_DATA, batch_size);
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network(XOR);
    Scalar log_loss = network.netTrain.log_loss();
    Show show1 = new Show();
    {
      TableBuilder table = network.netParamTracker.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(table.getColumns(0, i)));
      show1.setShowLabel("Log[Loss] = " + log_loss.maps(Round._3));
    }
    Show show2 = new Show();
    {
      TableBuilder table = network.netTrain.tloss;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show2.add(ListLinePlot.of(table.getColumns(0, i)));
      show2.setShowLabel("Log[Loss] = " + log_loss.maps(Round._3));
    }
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    new XORNeuralNetwork().runStandalone();
  }
}
