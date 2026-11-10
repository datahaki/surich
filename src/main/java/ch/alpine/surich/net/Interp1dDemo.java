// code by jph
package ch.alpine.surich.net;

import java.awt.BasicStroke;
import java.awt.Container;
import java.awt.Stroke;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.ElementwiseLayer;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.api.ScalarTensorFunction;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.itp.BSplineFunctionString;
import ch.alpine.tensor.itp.LinearInterpolation;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Interp1dDemo implements ManipulateProvider {
  private static final Stroke STROKE = //
      new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[] { 3 }, 0);
  public Integer dom = 10;
  public Integer samples = 25;
  public Integer hidden = 16;
  public Integer batch_size = 5;
  public Integer maxEpoch = 1000;
  public Scalar learningRate = RealScalar.of(0.1);
  public Scalar timeout = Quantity.of(0.5, "s");
  @FieldFuse
  public transient Boolean retrain = false;
  Tensor xdata;
  Tensor ydata;

  class NetLayerWrap {
    final NetLayer netLayer;
    final NetTrain netTrain;

    public NetLayerWrap(NetLayer netLayer) {
      this.netLayer = netLayer;
      netLayer.initialize(ThreadLocalRandom.current());
      netTrain = new NetTrain(netLayer, xdata, ydata);
      netTrain.runAsBatches(learningRate, batch_size, maxEpoch, timeout);
    }
  }

  @Override
  public Container getContainer() {
    Clip domain = Clips.absoluteOne();
    Tensor disY = RandomVariate.of(NormalDistribution.standard(), dom);
    ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
    ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(dom - 1));
    xdata = RandomVariate.of(UniformDistribution.of(domain), samples, 1);
    ScalarUnaryOperator suo = s -> (Scalar) stf.apply(lin.apply(domain.rescale(s)));
    ydata = xdata.maps(suo);
    // ---
    NetLayerWrap netLayerWrap0 = new NetLayerWrap(NetChains.sequence( //
        () -> ElementwiseLayer.gelu(), 1, hidden, hidden, hidden, 1));
    NetLayerWrap netLayerWrap1 = new NetLayerWrap(NetChains.sequence( //
        () -> ElementwiseLayer.tanh(), 1, hidden, hidden, hidden, 1));
    Show showF = new Show();
    {
      showF.setShowLabel("function");
      Showable showable = showF.add(Plot.of(suo, domain));
      showable.setStroke(STROKE);
      showF.add(Plot.of(NetLayers.asSUO11(netLayerWrap0.netLayer), domain)).setLabel("gelu");
      showF.add(Plot.of(NetLayers.asSUO11(netLayerWrap1.netLayer), domain)).setLabel("tanh");
      showF.add(ListPlot.of(Join.of(1, xdata, ydata)));
    }
    Show show0 = new Show();
    {
      show0.setShowLabel("net param");
      TableBuilder table = netLayerWrap0.netTrain.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show0.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    Show show1 = new Show();
    {
      show1.setShowLabel("loss");
      TableBuilder table = netLayerWrap0.netTrain.tloss;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show1.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    return ShowGridComponent.of(showF, show0, show1);
    // return new JLabel();
  }

  static void main() {
    new Interp1dDemo().runStandalone();
    // Path user_dir = Path.of(System.getProperty("user.dir"));
    // IO.println(user_dir);
  }
}
