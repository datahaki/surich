// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.subare.net.ScalarILayer;
import ch.alpine.subare.net.ScalarOLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
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
  public Integer hidden = 10;
  public Integer samples = 10;
  public Integer maxEpoch = 1000;
  public Scalar learningRate = RealScalar.of(0.1);

  @Override
  public Container getContainer() {
    Clip domain = Clips.absoluteOne();
    Tensor disY = RandomVariate.of(NormalDistribution.standard(), samples);
    ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
    ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(samples - 1));
    Tensor xdata = RandomVariate.of(UniformDistribution.of(domain), samples);
    ScalarUnaryOperator suo = s -> (Scalar) stf.apply(lin.apply(domain.rescale(s)));
    Tensor ydata = xdata.maps(suo);
    // ---
    NetLayer netLayer = NetChain.of( //
        new ScalarILayer(), //
        NetChains.feedForward(1, hidden, 1), //
        new ScalarOLayer() //
    );
    netLayer.initialize(ThreadLocalRandom.current());
    NetTrain netTrain = new NetTrain(netLayer, xdata, ydata);
    netTrain.run(learningRate, Quantity.of(1, "s"), maxEpoch, 1);
    Show showF = new Show();
    {
      showF.setShowLabel("function");
      showF.add(Plot.of(suo, domain));
      showF.add(Plot.of(NetLayers.asSUO(netLayer), domain));
      showF.add(ListPlot.of(xdata, ydata));
    }
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
    return ShowGridComponent.of(showF, show0, show1);
  }

  static void main() {
    new Interp1dDemo().runStandalone();
  }
}
