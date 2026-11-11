// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.LinearLayer;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
class LinearLayerDemo implements ManipulateProvider {
  public Integer d_in = 1;
  public Integer d_out = 1;
  public Integer samples = 20;
  public Integer batch_size = 5;
  public Scalar learningRate = RealScalar.of(0.01);

  @Override
  public Container getContainer() {
    Distribution distribution = NormalDistribution.standard();
    TensorUnaryOperator tuo;
    {
      NetLayer netLayer = NetChain.of(new LinearLayer(d_in, d_out));
      netLayer.initialize(ThreadLocalRandom.current());
      tuo = netLayer::forward;
    }
    Tensor xdata = RandomVariate.of(distribution, samples, d_in);
    Tensor ydata = tuo.slash(xdata);
    // ---
    NetLayer netLayer = NetChain.of(new LinearLayer(d_in, d_out));
    netLayer.initialize(ThreadLocalRandom.current());
    NetTrain netTrain = new NetTrain(netLayer, xdata, ydata);
    netTrain.runAsBatches(learningRate, batch_size, 100, Quantity.of(1, "s"));
    Show show0 = new Show();
    {
      show0.setShowLabel("net param");
      TableBuilder table = netTrain.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show0.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    Show show1 = new Show();
    {
      show1.setShowLabel("loss");
      TableBuilder table = netTrain.tloss;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    return ShowGridComponent.of(show0, show1);
  }

  static void main() {
    new LinearLayerDemo().runStandalone();
  }
}
