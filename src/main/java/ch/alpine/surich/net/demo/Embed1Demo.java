// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.ScalarEmbeddingLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.UnitStep;

@ReflectionMarker
class Embed1Demo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new Embed1Component();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new Embed1Demo().runStandalone();
  }
}

@ReflectionMarker
class Embed1Component implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer V = 30;
  public Integer D = 3;
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(Clips.absolute(2), Clips.absolute(2));
    final TrainingData trainingData;

    public InnerComponent() {
      super(0);
      Tensor value = Range.of(0, V);
      Tensor xdata = value.maps(Tensors::of);
      Tensor ydata = value.maps(v -> Boole.of(v.number().intValue() % 2 == 0));
      // assignment of random labels to points
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetModel netModel = NetModels.logits(NetChain.join( //
          new ScalarEmbeddingLayer(V, D), //
          optLinearChain.create(D, 1)));
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer, randomGenerator);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Show show = new Show("even");
      Tensor X = trainingData.xdata();
      Tensor Y = netLayer.forwardBatch(X).maps(UnitStep.FUNCTION);
      show.add(ListDiscretePlot.of(Flatten.of(X), Y));
      show.render_autoIndent(graphics, list.get(0));
    }
  }
}
