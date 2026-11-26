// code by jph
package ch.alpine.surich.net.demo;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelDataBatch;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Tuples;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Mod;

@ReflectionMarker
class XorDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new XorComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new XorDemo().runStandalone();
  }
}

@ReflectionMarker
class XorComponent implements Supplier<NetTrainComponent> {
  public Integer seed = 0;
  public Integer batch_size = 4;
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public final OptParam optParam = new OptParam();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  public static Scalar xor(Tensor xy) {
    return Mod.function(2).apply(Total.ofVector(xy));
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      Tensor X = Tuples.of(Range.of(0, 2), 2);
      Tensor Y = Slash.of(X, XorComponent::xor);
      trainingData = new TrainingData(X, Y);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetModelDataBatch createNetTrainDataBatch() {
      NetModel netModel = NetModels.logits(optLinearChain.create(2, 1), ScalarLayer.VectorToScalar);
      return new NetModelDataBatch(netModel, trainingData, batch_size);
    }
  }
}
