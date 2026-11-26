// code by jph
package ch.alpine.surich.net.pen;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.Flatten1Layer;
import ch.alpine.subare.net.lay.IdentityLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.lay.SlicedLayer;
import ch.alpine.subare.net.lay.TakeListLayer;
import ch.alpine.subare.net.lay.UnitVectorLayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.DatasetBuilder;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.nrm.Vector2Norm;

@ReflectionMarker
class PenguinSpxDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new PenguinSpxComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new PenguinSpxDemo().runStandalone();
  }
}

@ReflectionMarker
class PenguinSpxComponent extends NetTrainComponentSupplier {
  private final DatasetBuilder datasetBuilder = PenguinDataset.INSTANCE.builder;
  public final OptLinearChain optLinearChain = new OptLinearChain();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      // assignment of random labels to points
      trainingData = new TrainingData( //
          datasetBuilder.extract(1, 2, 3, 4, 6), //
          datasetBuilder.extract(0, 5)); // species and mass
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      int vi = datasetBuilder.distinctCount(1); // island
      NetLayer fi = NetChain.join(SingletonLayer.VectorToScalar, new UnitVectorLayer(vi));
      NetLayer fn = IdentityLayer.INSTANCE;
      NetLayer netLayer = NetChain.join( //
          new TakeListLayer(1, 4), //
          SlicedLayer.of(fi, fn), //
          new Flatten1Layer(), //
          optLinearChain.create(vi + 4, 4), //
          new TakeListLayer(3, 1));
      // TODO bad API design
      NetModel m1 = NetModels.softmax_argMax(IdentityLayer.INSTANCE);
      NetModel m2 = new NetModel(SingletonLayer.VectorToScalar, SquaredErrorLoss.INSTANCE);
      NetModel netModel = NetModels.output(netLayer, List.of(m1, m2));
      return new NetModelData(netModel, trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Tensor xdata = trainingData.xdata();
      Tensor ydata = trainingData.ydata();
      Tensor ypred = Slash.of(xdata, netLayer::forward);
      Tensor x = Range.of(0, xdata.length());
      Tensor y = Slash.of(ypred.subtract(ydata), Vector2Norm::of);
      Show show = new Show("error");
      show.add(ListDiscretePlot.of(x, y));
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
