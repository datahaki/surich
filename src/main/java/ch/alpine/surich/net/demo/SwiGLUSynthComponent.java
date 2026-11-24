// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.err.SquaredErrorLoss;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class SwiGLUSynthComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 200;
  public Integer dim0 = 3;
  public Integer dim1 = 5;
  public Integer dim2 = 2;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 20;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  private NetLayer create() {
    SwiGLULayer n1 = new SwiGLULayer(dim0, dim1, dim2);
    n1.netRegularization = NoRegularization.INSTANCE;
    return n1;
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    NetTrain netTrain;

    public InnerComponent() {
      super(seed);
      NetLayer netLayer = create();
      NetLayers.initialize(netLayer, randomGenerator);
      Tensor xdata = RandomVariate.of(UniformDistribution.of(-1, 1), randomGenerator, samples, dim0);
      Tensor ydata = netLayer.forwardBatch(xdata);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer netLayer = create();
      NetLayers.initialize(netLayer, randomGenerator);
      netTrain = new NetTrain( //
          new NetModel(netLayer, SquaredErrorLoss.INSTANCE), //
          optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(3);
      {
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        NetShows.pMatrix(netLayer).render_autoIndent(graphics, list.get(3));
      }
    };
  }
}
