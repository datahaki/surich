// code by jph
package ch.alpine.surich.net.pen;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.math.Unitless;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer;
import ch.alpine.subare.net.lay.Flatten1Layer;
import ch.alpine.subare.net.lay.IdentityLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.OneHotLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.lay.TakeListLayer;
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
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Round;

@ReflectionMarker
class PenguinSexDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new PenguinSexComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new PenguinSexDemo().runStandalone();
  }
}

@ReflectionMarker
class PenguinSexComponent extends NetTrainComponentSupplier {
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
          datasetBuilder.extract(0, 1, 2, 3, 4, 5), //
          datasetBuilder.single(6));
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      int vs = datasetBuilder.distinctCount(0); // species
      int vi = datasetBuilder.distinctCount(1); // island
      NetLayer fs = NetChain.join(SingletonLayer.VectorToScalar, new OneHotLayer(vs));
      NetLayer fi = NetChain.join(SingletonLayer.VectorToScalar, new OneHotLayer(vi));
      NetLayer fn = IdentityLayer.INSTANCE;
      NetModel netModel = NetModels.logits( //
          new TakeListLayer(1, 1, 4), //
          FeatureEmbeddingLayer.of(fs, fi, fn), //
          new Flatten1Layer(), //
          optLinearChain.create(vs + vi + 4, 1), //
          SingletonLayer.VectorToScalar);
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
      Tensor y = Entrywise.neq().apply(ypred, ydata);
      Tensor eperc = Unitless.PERCENT.unitConvert().apply(Mean.ofVector(y)).maps(Round._1);
      Show show = new Show("error " + eperc);
      show.add(ListDiscretePlot.of(x, y));
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
