// code by jph
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.grd.AxisMaps;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetParamTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.api.NetOptimizer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.ResidualLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

@ReflectionMarker
class ResidualDemo implements ManipulateProvider {
  public Integer dinout = 4;
  public Integer hidden = 10;
  public Integer samples = 20;
  public Integer batch_size = 5;
  public Integer maxEpoch = 200;
  public final OptParam optParam = new OptParam();

  @Override
  public Container getContainer() {
    NetLayer netLayer = NetChains.argMax( //
        new ResidualLayer(NetChains.linear(ElementwiseLayers.GELU, dinout, hidden, dinout)));
    NetInitialize.of(netLayer);
    Tensor xdata = RandomVariate.of(NormalDistribution.standard(), samples, dinout);
    TensorUnaryOperator tuo = netLayer::forward;
    Tensor ydata = tuo.slash(xdata);
    // ---
    NetInitialize.of(netLayer);
    NetOptimizer netOptimize = optParam.create(netLayer);
    NetTrain netTrain = new NetTrain(netLayer, netOptimize);
    NetParamTracker netParamTracker = new NetParamTracker();
    netTrain.addListener(netParamTracker);
    TrainingData trainingData = new TrainingData(xdata, ydata);
    netTrain.setMaxEpochs(maxEpoch);
    netTrain.runAsBatches(trainingData, batch_size);
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
      show1.showOptions.axisOptionsY.axisMap = AxisMaps.LOG;
    }
    return ShowGridComponent.of(show0, show1);
  }

  static void main() {
    new ResidualDemo().runStandalone();
  }
}
