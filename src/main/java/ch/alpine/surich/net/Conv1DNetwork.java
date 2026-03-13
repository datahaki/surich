// code adapted from chatgpt
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.nrm.FrobeniusNorm;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Round;

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
  // ---
  // public Scalar l2 = RealScalar.of(0.00001);
  public Scalar learningRate = RealScalar.of(0.1);
  public Integer maxEpoch = 100;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private static final int SKIP = 2;
    private final NetChain netChain = NetChains.cnnRelu(3);
    private final NetTrain netTrain;
    Scalar error;

    /** @param Y */
    Network(Tensor Y) {
      // netChain.setL2(l2);
      netTrain = new NetTrain(netChain, X, Y);
      netTrain.run(learningRate, timeout, maxEpoch, SKIP);
      // ---
      error = FrobeniusNorm.of(netTrain.error());
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network(Y);
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
    CONV1DTOY.check(X, Y);
    new Conv1DNetwork().runStandalone();
  }
}
