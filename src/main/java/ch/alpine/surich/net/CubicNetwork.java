// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.function.Consumer;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Plot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldSelectionArray;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetChains;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.nrm.VectorInfinityNorm;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.ArcSinDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Round;
import ch.alpine.tensor.sca.pow.Power;

@ReflectionMarker
public class CubicNetwork implements ManipulateProvider {
  public static final Clip clip = Clips.absolute(1);
  public static final int SKIP = 10;
  public Scalar scale = RealScalar.of(2);
  @FieldSelectionArray({ "2", "3", "4", "6", "7", "8", "10", "12" })
  public Integer hiddenSize = 4;
  public Scalar l2 = RealScalar.of(1e-5);
  public Scalar learningRate = RealScalar.of(0.02);
  public Integer maxEpoch = 4001;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private final ScalarUnaryOperator power = s -> Power.function(3).apply(s.multiply(scale));
    private final NetChain netChain = NetChains.linTanhLin(1, hiddenSize, 1);
    private final TableBuilder tableBuilder = new TableBuilder();

    void train() {
      netChain.setL2(l2);
      Consumer<Tensor> consumer = tableBuilder::appendRow;
      int epoch = 0;
      Timing timing = Timing.started();
      Distribution distribution = ArcSinDistribution.INSTANCE;
      while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
        Tensor x = Tensors.of(RandomVariate.of(distribution));
        Tensor y = netChain.forward(x);
        //
        Tensor diff = x.maps(power).subtract(y).multiply(learningRate);
        // Tensor loss = Entrywise.mul().apply(diff, diff);
        Tensor d = diff.add(diff);
        //
        netChain.back(d);
        netChain.update();
        if (epoch % SKIP == 0)
          consumer.accept(netChain.parameters());
        ++epoch;
      }
    }

    Scalar evaluate() {
      Tensor errors = Tensors.empty();
      for (Tensor x : Subdivide.increasing(clip, 20).maps(Tensors::of)) {
        Tensor y = netChain.forward(x);
        Tensor t = x.maps(power);
        Tensor e = t.subtract(y).Get(0);
        errors.append(e);
      }
      return VectorInfinityNorm.of(errors);
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network();
    network.train();
    Scalar error = network.evaluate();
    Tensor table = network.tableBuilder.getTable();
    IO.println("Error: " + error);
    int n = Unprotect.dimension1Hint(table);
    Tensor domain = Range.of(0, table.length()).multiply(RealScalar.of(SKIP));
    Show show1 = new Show();
    for (int i = 0; i < n; ++i)
      show1.add(ListLinePlot.of(domain, table.get(Tensor.ALL, i)));
    Show show2 = new Show();
    show2.setPlotLabel("Infty error: " + error.maps(Round._3));
    show2.add(Plot.of(network.power, clip)).setLabel("cubic");
    show2.add(Plot.of(x -> network.netChain.forward(Tensors.of(x)).Get(0), clip)).setLabel("network");
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    new CubicNetwork().runStandalone();
  }
}
