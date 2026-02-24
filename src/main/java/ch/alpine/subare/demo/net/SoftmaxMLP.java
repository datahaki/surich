// code adapted from chatgpt
package ch.alpine.subare.demo.net;

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
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.nrm.FrobeniusNorm;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
public class SoftmaxMLP implements ManipulateProvider {
  static final Distribution DISTRIBUTION = NormalDistribution.of(0.0, 0.1);
  public static final Tensor X = Tensors.matrixDouble(new double[][] { //
      { 1, 1 }, { 1.5, 2 }, { 2, 1 }, // Class 0
      { 5, 5 }, { 6, 5 }, { 5, 6 }, // Class 1
      { 8, 1 }, { 9, 2 }, { 8, 2 } // Class 2
  }).unmodifiable();
  public static final Tensor y = Tensors.vectorInt(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 }).unmodifiable();
  public static final int SKIP = 10;
  @FieldSelectionArray({ "6", "7", "8", "10" })
  public Integer hiddenSize = 8;
  public Scalar l2 = RealScalar.of(1e-5);
  public Scalar learningRate = RealScalar.of(0.05);
  public Integer maxEpoch = 8000;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private final NetChain netChain = NetChains.argMaxMLP(2, hiddenSize, 3);
    private final TableBuilder tableBuilder = new TableBuilder();

    void train(Tensor x, Tensor y) {
      netChain.setL2(l2);
      NetTrain.of(netChain, x, y, learningRate, tableBuilder::appendRow, timeout, maxEpoch, SKIP);
    }

    Scalar evaluate(Tensor X, Tensor T) {
      System.out.println("Evaluation:");
      Tensor errors = Tensors.empty();
      for (int n = 0; n < X.length(); n++) {
        Tensor x = X.get(n);
        Tensor y = netChain.forward(x);
        Tensor error = y.subtract(T.get(n));
        System.out.println("I: " + x + " | " + y + "=" + T.get(n)); // + probs.maps(Round._2)
        errors.append(error);
      }
      return FrobeniusNorm.of(errors);
    }
  }

  public Show getShow() {
    Network network = new Network();
    network.train(X, y);
    Scalar error = network.evaluate(X, y);
    Tensor table = network.tableBuilder.getTable();
    IO.println(error);
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
    new SoftmaxMLP().runStandalone();
  }
}
