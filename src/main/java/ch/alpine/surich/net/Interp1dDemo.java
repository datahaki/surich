// code by jph
package ch.alpine.surich.net;

import java.awt.BasicStroke;
import java.awt.Container;
import java.awt.Stroke;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamTracker;
import ch.alpine.subare.net.NetWrap;
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
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Interp1dDemo implements ManipulateProvider {
  private static final Stroke STROKE = //
      new BasicStroke(1.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[] { 3 }, 0);
  public final Interp1Param interp1Param = new Interp1Param();
  public Integer hidden = 7;
  public Scalar beta = RealScalar.of(0.9);
  public Scalar eta = RealScalar.of(0.03);
  public Integer batch_size = 6;
  public Integer maxEpoch = 1000;
  public Scalar timeout = Quantity.of(1, "s");
  @FieldFuse
  public transient Boolean retrain = false;

  class NetLayerWrap {
    final NetLayer netLayer;
    final NetTrain netTrain;
    final NetParamTracker netParamTracker = new NetParamTracker();

    public NetLayerWrap(NetLayer netLayer, TrainingData trainingdata) {
      this.netLayer = netLayer;
      NetInitialize.of(netLayer);
      NetOptimizer netOptimize = new NetMomentumGradientDescent(netLayer, beta, eta);
      netTrain = new NetTrain(netLayer, netOptimize);
      netTrain.addListener(netParamTracker);
      netTrain.runAsBatches(trainingdata, batch_size, maxEpoch, timeout);
    }
  }

  @Override
  public Container getContainer() {
    TrainingData trainingdata = interp1Param.generate();
    Tensor xdata = trainingdata.xdata();
    Tensor ydata = trainingdata.ydata();
    // ---
    NetLayerWrap netLayerWrap0 = new NetLayerWrap(NetChains.linear( //
        () -> ElementwiseLayer.gelu(), 1, hidden, hidden, hidden, 1), trainingdata);
    NetLayerWrap netLayerWrap1 = new NetLayerWrap(NetChains.linear( //
        () -> ElementwiseLayer.tanh(), 1, hidden, hidden, hidden, 1), trainingdata);
    Show showF = new Show();
    {
      Clip domain = Clips.absoluteOne();
      showF.setShowLabel("function");
      // Showable showable = showF.add(Plot.of(suo, domain));
      // showable.setLabel("truth");
      // showable.setStroke(STROKE);
      showF.add(Plot.of(NetWrap.asSUO11(netLayerWrap0.netLayer), domain)).setLabel("gelu");
      showF.add(Plot.of(NetWrap.asSUO11(netLayerWrap1.netLayer), domain)).setLabel("tanh");
      showF.add(ListPlot.of(Join.of(1, xdata, ydata)));
    }
    Show show0 = new Show();
    {
      show0.setShowLabel("net param");
      TableBuilder table = netLayerWrap0.netParamTracker.tparam;
      int n = table.getRow(0).length();
      for (int i = 1; i < n; ++i)
        show0.add(ListLinePlot.of(table.getColumns(0, i)));
    }
    Show show1 = new Show();
    {
      show1.setShowLabel("loss");
      show1.add(ListLinePlot.of(netLayerWrap0.netTrain.tloss.getTable())).setLabel("gelu");
      show1.add(ListLinePlot.of(netLayerWrap1.netTrain.tloss.getTable())).setLabel("tanh");
    }
    return ShowGridComponent.of(showF, show0, show1);
  }

  static void main() {
    new Interp1dDemo().runStandalone();
  }
}
