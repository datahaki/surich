// code by jph
package ch.alpine.surich.net.demo;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class SwiGLUSynthDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new SwiGLUSynthComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new SwiGLUSynthDemo().runStandalone();
  }
}

@ReflectionMarker
class SwiGLUSynthComponent extends NetTrainComponentSupplier {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer samples = 200;
  public Integer dim0 = 3;
  public Integer dim1 = 5;
  public Integer dim2 = 2;

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
    public NetModelData createNetModelDataBatch() {
      NetLayer netLayer = create();
      return new NetModelData(new NetModel(netLayer, SquaredErrorLoss.INSTANCE), trainingData);
    }
  }
}
