// code adapted from chatgpt
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetSimpleGradientDescent;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

/** Quote from chatgpt:
 * 
 * Also uses sigmoid activation */
@ReflectionMarker
class Conv1DNetwork implements ManipulateProvider {
  // public static final Tensor X = Tensors.of(Tensors.vectorDouble(0, 0, 1, 1)).unmodifiable();
  // public static final Tensor Y = Tensors.of(Tensors.vectorDouble(0, 1)).unmodifiable();
  public static final Conv1DToy CONV1DTOY = new Conv1DToy(3);
  public static final Tensor X = RandomVariate.of(NormalDistribution.standard(), 20, 10);
  public static final Tensor Y = CONV1DTOY.target(X);
  public static final TrainingData TRAINING_DATA = new TrainingData(X, Y);
  // ---
  // public Scalar l2 = RealScalar.of(0.00001);
  public Scalar learningRate = RealScalar.of(0.1);
  public Integer batch_size = 100;
  public Integer maxEpoch = 100;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private final NetLayer netChain = NetChains.cnnRelu(3);
    private final NetParamTracker netParamTracker = new NetParamTracker();
    private final NetTrain netTrain;

    /** @param Y */
    Network(Tensor Y) {
      // netChain.setL2(l2);
      NetOptimizer netOptimize = new NetSimpleGradientDescent(netChain, learningRate);
      netTrain = new NetTrain(netChain, netOptimize);
      netTrain.addListener(netParamTracker);
      netTrain.setMaxEpochs(maxEpoch);
      netTrain.runAsBatches(TRAINING_DATA, batch_size);
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network(Y);
    Show show1 = new Show();
    {
      TableBuilder table = network.netParamTracker.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    Show show2 = new Show();
    {
      TableBuilder table = network.netTrain.tloss;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show2.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    CONV1DTOY.check(X, Y);
    new Conv1DNetwork().runStandalone();
  }
}
