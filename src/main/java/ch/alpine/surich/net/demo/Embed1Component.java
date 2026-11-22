// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.GraphicsGrid;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.ScalarEmbeddingLayer;
import ch.alpine.subare.net.run.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Embed1Component implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer V = 30;
  public Integer D = 3;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(Clips.absolute(2), Clips.absolute(2));
    final TrainingData trainingData;
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public InnerComponent() {
      Tensor xdata = Range.of(0, V).maps(Tensors::of);
      Tensor ydata = xdata.maps(v -> Boole.of(v.number().intValue() % 2 == 0));
      // assignment of random labels to points
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer inner = NetChains.linear(between, D, 16, 8, 1);
      NetLayer netLayer = NetChains.binary(NetChain.join(new ScalarEmbeddingLayer(V, D), inner));
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain(netLayer, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
      NetLayer copy = InnerComponent.this.copy;
      if (Objects.nonNull(copy)) {
        List<Rectangle> list = GraphicsGrid.rectangles(3, getSize());
        {
          Show show = new Show("even");
          Tensor X = trainingData.xdata();
          Tensor Y = copy.forwardBatch(X);
          show.add(ListDiscretePlot.of(Flatten.of(X), Flatten.of(Y)));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
          NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(3));
        }
      }
    }
  }
}
