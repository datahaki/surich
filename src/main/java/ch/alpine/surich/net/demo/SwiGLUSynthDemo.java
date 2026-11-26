// code by jph
package ch.alpine.surich.net.demo;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class SwiGLUSynthDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new SwiGLUSynthComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new SwiGLUSynthDemo().runStandalone();
  }
}

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
      netTrain = new NetTrain( //
          new NetModel(netLayer, SquaredErrorLoss.INSTANCE), //
          optParam.create());
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }
  }
}
