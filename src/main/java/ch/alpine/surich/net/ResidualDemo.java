// code by jph
package ch.alpine.surich.net;

import java.awt.Container;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.ResidualLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.run.NetParamTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.tensor.Tensor;
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
    NetModel netModel = NetModels.argMax( //
        new ResidualLayer(NetChains.linear(ElementwiseLayers.GELU, dinout, hidden, dinout)));
    NetLayer netLayer = netModel.netLayer();
    NetLayers.initialize(netLayer);
    Tensor xdata = RandomVariate.of(NormalDistribution.standard(), samples, dinout);
    Tensor ydata = netLayer.forwardBatch(xdata);
    // ---
    NetLayers.initialize(netLayer);
    NetOptimizer netOptimize = optParam.create(netLayer);
    NetTrain netTrain = new NetTrain(netModel, netOptimize);
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
    Show show1 = NetShows.loss(netTrain.tloss);
    return ShowGridComponent.of(show0, show1);
  }

  static void main() {
    new ResidualDemo().runStandalone();
  }
}
