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
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Round;

@ReflectionMarker
class SoftmaxMLP implements ManipulateProvider {
  static final Distribution DISTRIBUTION = NormalDistribution.of(0.0, 0.1);
  public static final Tensor X = Tensors.matrixDouble(new double[][] { //
      { 1, 1 }, { 1.5, 2 }, { 2, 1 }, // Class 0
      { 5, 5 }, { 6, 5 }, { 5, 6 }, // Class 1
      { 8, 1 }, { 9, 2 }, { 8, 2 } // Class 2
  }).unmodifiable();
  public static final Tensor Y = Tensors.vectorInt(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 }).unmodifiable();
  public static final int SKIP = 10;
  @FieldSelectionArray({ "6", "7", "8", "10" })
  public Integer hiddenSize = 8;
  public Scalar l2 = RealScalar.of(1e-5);
  public Scalar learningRate = RealScalar.of(0.05);
  public Integer maxEpoch = 8000;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private final NetChain netChain = NetChains.argMaxMLP(2, hiddenSize, 3);
    private final NetTrain netTrain;
    Scalar error;

    Network(Tensor X, Tensor Y) {
      netChain.setL2(l2);
      netTrain = new NetTrain(netChain, X, Y);
      netTrain.run(learningRate, timeout, maxEpoch, SKIP);
      // ---
      error = FrobeniusNorm.of(netTrain.error());
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network(X, Y);
    Scalar error = network.error;
    Show show1 = new Show();
    {
      TableBuilder tableBuilder = network.netTrain.tparam;
      int n = tableBuilder.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(tableBuilder.getColumns(0, i)));
      show1.setShowLabel("Error: " + error.maps(Round._3));
    }
    Show show2 = new Show();
    {
      TableBuilder tableBuilder = network.netTrain.tloss;
      int n = tableBuilder.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show2.add(ListLinePlot.of(tableBuilder.getColumns(0, i)));
      show2.setShowLabel("Error: " + error.maps(Round._3));
    }
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    new SoftmaxMLP().runStandalone();
  }
}
