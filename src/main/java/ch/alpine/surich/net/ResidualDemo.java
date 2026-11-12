// code by jph
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.ArgMaxLayer;
import ch.alpine.subare.net.lay.ElementwiseLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ResidualLayer;
import ch.alpine.subare.net.lay.SoftmaxLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetSimpleGradientDescent;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
class ResidualDemo implements ManipulateProvider {
  public Integer dinout = 4;
  public Integer hidden = 10;
  public Integer samples = 20;
  public Integer batch_size = 5;
  public Integer maxEpoch = 200;
  public Scalar learningRate = RealScalar.of(0.1);

  @Override
  public Container getContainer() {
    NetLayer netLayer = NetChain.of( //
        new ResidualLayer(NetChains.linear(() -> ElementwiseLayer.gelu(), dinout, hidden, dinout)), //
        SoftmaxLayer.INSTANCE, //
        new ArgMaxLayer());
    NetInitialize.of(netLayer);
    Tensor xdata = RandomVariate.of(NormalDistribution.standard(), samples, dinout);
    TensorUnaryOperator tuo = netLayer::forward;
    Tensor ydata = tuo.slash(xdata);
    // ---
    NetInitialize.of(netLayer);
    NetOptimizer netOptimize = new NetSimpleGradientDescent(netLayer,learningRate);
    NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    NetParamTracker netParamTracker = new NetParamTracker();
    netTrain.addListener(netParamTracker);
    TrainingData trainingData = new TrainingData(xdata, ydata);
    netTrain.runAsBatches(trainingData, batch_size, maxEpoch, Quantity.of(1, "s"));
    Show show0 = new Show();
    {
      show0.setShowLabel("net param");
      TableBuilder table = netParamTracker.tparam;
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
    new ResidualDemo().runStandalone();
  }
}
