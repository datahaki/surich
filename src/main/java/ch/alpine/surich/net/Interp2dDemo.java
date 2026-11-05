// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Interp2dDemo implements ManipulateProvider {
  public Integer hidden = 10;
  public Integer samples = 10;
  public Integer maxEpoch = 1000;
  public Scalar learningRate = RealScalar.of(0.1);
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
      netTrain.run(learningRate, Quantity.of(1, "s"), maxEpoch, 10);
    }
  }

  @Override
  public Container getContainer() {
    Clip domain = Clips.absoluteOne();
    // Tensor array = Get.of(HomeDirectory.Ephemeral.resolve("interp2d.tensor"));
    // Tensor disY = RandomVariate.of(NormalDistribution.standard(), samples,2);
    // ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
    // ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(samples - 1));
    // xdata = RandomVariate.of(UniformDistribution.of(domain), samples);
    // ScalarUnaryOperator suo = s -> (Scalar) stf.apply(lin.apply(domain.rescale(s)));
    // ydata = xdata.maps(suo);
    // // ---
    // NetLayerWrap netLayerWrap0 = new NetLayerWrap(NetChain.of( //
    // NetChains.feedForward(2, hidden, 1), //
    // new ScalarOLayer() //
    // ));
    // Show showF = new Show();
    // {
    // showF.setShowLabel("function");
    // Showable showable = showF.add(Plot.of(suo, domain));
    // showF.add(Plot.of(NetLayers.asSUO(netLayerWrap0.netLayer), domain)).setLabel("gelu");
    // showF.add(Plot.of(NetLayers.asSUO(netLayerWrap1.netLayer), domain)).setLabel("tanh");
    // showF.add(ListPlot.of(xdata, ydata));
    // }
    // Show show0 = new Show();
    // {
    // show0.setShowLabel("net param");
    // TableBuilder table = netLayerWrap0.netTrain.tparam;
    // int n = table.getRow(0).length();
    // for (int i = 1; i < n; ++i)
    // show0.add(ListLinePlot.of(table.getColumns(0, i)));
    // }
    // Show show1 = new Show();
    // {
    // show1.setShowLabel("loss");
    // TableBuilder table = netLayerWrap0.netTrain.tloss;
    // int n = table.getRow(0).length();
    // for (int i = 1; i < n; ++i)
    // show1.add(ListLinePlot.of(table.getColumns(0, i)));
    // }
    // return ShowGridComponent.of(showF, show0, show1);
    return null;
  }

  static void main() {
    new Interp2dDemo().runStandalone();
  }
}
