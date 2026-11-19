// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.function.Consumer;
import java.util.stream.Stream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldSelectionArray;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetExtract;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.ArcSinDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.pow.Power;

@ReflectionMarker
class CubicNetwork implements ManipulateProvider {
  public static final Clip CLIP = Clips.absolute(1);
  public Scalar scale = RealScalar.of(2);
  @FieldSelectionArray({ "2", "3", "4", "6", "7", "8", "10", "12" })
  public Integer hiddenSize = 4;
  public Scalar l2 = RealScalar.of(1e-5);
  public Scalar learningRate = RealScalar.of(0.01);
  public Integer batch_size = 10;
  public Integer maxEpoch = 4001;
  public Scalar timeout = Quantity.of(1, "s");

  public class Network {
    private static final int SKIP = 10;
    private final ScalarUnaryOperator power = s -> Power.function(3).apply(s.multiply(scale));
    private final NetLayer netLayer = NetChains.linear(ElementwiseLayers.TANH, 1, hiddenSize, 1);
    private final TableBuilder tableBuilder = new TableBuilder();
    private TableBuilder vloss;

    void train() {
      NetInitialize.of(netLayer);
      NetOptimizer netOptimizer = NetOptimizers.SIMPLE.create(netLayer, learningRate);
      NetTrain netTrain = new NetTrain(netLayer, netOptimizer);
      Consumer<Tensor> consumer = tableBuilder::appendRow;
      int epoch = 0;
      Timing timing = Timing.started();
      Distribution distribution = ArcSinDistribution.INSTANCE;
      while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
        Tensor xdata = RandomVariate.of(distribution, batch_size, 1);
        Tensor ydata = xdata.maps(power);
        netTrain.setMaxEpochs(1000);
        netTrain.runAsBatches(new TrainingData(xdata, ydata), batch_size);
        if (epoch % SKIP == 0)
          consumer.accept(Tensor.of(Stream.concat(Stream.of(RealScalar.of(epoch)), new NetExtract(netLayer).all())));
        ++epoch;
      }
      vloss = netTrain.tloss;
    }
  }

  @Override
  public Container getContainer() {
    Network network = new Network();
    network.train();
    // Scalar error = network.evaluate();
    Show show0 = new Show();
    { // show2.setShowLabel("Infty error: " + error.maps(Round._3));
      show0.add(Plot.of(network.power, CLIP)).setLabel("cubic");
      show0.add(Plot.of(x -> network.netLayer.forward(Tensors.of(x)).Get(0), CLIP)).setLabel("network");
    }
    Show show1 = new Show();
    {
      TableBuilder tableBuilder = network.tableBuilder;
      int n = tableBuilder.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(tableBuilder.getColumns(0, i)));
    }
    Show show2 = new Show();
    {
      show2.add(ListLinePlot.of(network.vloss.getColumns(0, 1)));
    }
    return ShowGridComponent.of(show0, show1, show2);
  }

  static void main() {
    new CubicNetwork().runStandalone();
  }
}
