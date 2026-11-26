// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ArrayPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Partition;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Tuples;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Mod;

/** the smallest network is 2->Linear[2,1]->Tanh/Sigmoid->1 */
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
    public NetModelData createNetModelDataBatch() {
      NetModel netModel = NetModels.logits(optLinearChain.create(2, 1), SingletonLayer.VectorToScalar);
      return new NetModelData(netModel, trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel copy) {
      NetLayer netLayer = copy.netLayer();
      Tensor pred = Slash.of(trainingData.xdata(), netLayer::forward);
      Tensor array = Partition.of(pred, 2);
      List<Rectangle> rectangles = rectangles(1);
      Show show = new Show();
      show.add(ArrayPlot.of(array));
      show.render_autoIndent(graphics, rectangles.get(0));
    }
  }
}
