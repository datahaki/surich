// code by jph
package ch.alpine.surich.net.demo;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.MatrixLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.lay.ResidualLayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class ResidualSynthDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new ResidualSynthComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new ResidualSynthDemo().runStandalone();
  }
}

@ReflectionMarker
class ResidualSynthComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 200;
  public Integer batch_size = 20;
  public Integer dim0 = 6;
  public Integer dim1 = 10;
  public Integer dim2 = 3;
  public Scalar noise_sigma = RealScalar.of(0.0001);
  public final OptParam optParam = new OptParam();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  private NetLayer create() {
    MatrixLayer n1 = new LinearLayer(dim0, dim1);
    n1.netRegularization = NoRegularization.INSTANCE;
    MatrixLayer n2 = new MatrixLayer(dim1, dim0);
    n2.netRegularization = NoRegularization.INSTANCE;
    MatrixLayer n3 = new LinearLayer(dim0, dim2);
    n3.netRegularization = NoRegularization.INSTANCE;
    return NetChain.join(new ResidualLayer(NetChain.join(n1, n2)), n3);
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      NetLayer netLayer = create();
      NetLayers.initialize(netLayer, randomGenerator);
      Tensor xdata = RandomVariate.of(UniformDistribution.of(-1, 1), randomGenerator, samples, dim0);
      Tensor noise = RandomVariate.of(NormalDistribution.of(noise_sigma.zero(), noise_sigma), randomGenerator, samples, dim2);
      Tensor ydata = netLayer.forwardBatch(xdata).add(noise);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer netLayer = create();
      NetModel netModel = new NetModel(netLayer, SquaredErrorLoss.INSTANCE);
      netTrain = new NetTrain(netModel, optParam.create());
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }
  }
}
