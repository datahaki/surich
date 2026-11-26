// code by jph
package ch.alpine.surich.net.demo;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelDataBatch;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
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
  public final NetTrainComponentSupplier supplier = new XorComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new XorDemo().runStandalone();
  }
}

@ReflectionMarker
class XorComponent extends NetTrainComponentSupplier {
  public final OptLinearChain optLinearChain = new OptLinearChain();

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
    public NetModelDataBatch createNetModelDataBatch() {
      NetModel netModel = NetModels.logits(optLinearChain.create(2, 1), ScalarLayer.VectorToScalar);
      return new NetModelDataBatch(netModel, trainingData, batch_size);
    }
  }
}
