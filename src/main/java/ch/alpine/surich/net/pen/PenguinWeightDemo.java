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
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
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
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Round;

@ReflectionMarker
class PenguinWeightDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new PenguinWeightComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new PenguinWeightDemo().runStandalone();
  }
}

@ReflectionMarker
class PenguinWeightComponent extends NetTrainComponentSupplier {
  public static final Clip DOMAIN = Clips.absoluteOne();
  private final DatasetBuilder datasetBuilder = PenguinDataset.INSTANCE.builder;
  public final OptLinearChain optLinearChain = new OptLinearChain();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      // assignment of random labels to points
      trainingData = new TrainingData( //
          datasetBuilder.extract(0, 1, 2, 3, 4, 6), //
          datasetBuilder.single(5));
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      // using binary encoding, i.e. 0 or 1 for male/female
      int vs = datasetBuilder.distinctCount(0); // species
      int vi = datasetBuilder.distinctCount(1); // island
      NetLayer fs = NetChain.join(SingletonLayer.VectorToScalar, new OneHotLayer(vs));
      NetLayer fi = NetChain.join(SingletonLayer.VectorToScalar, new OneHotLayer(vi));
      NetLayer fn = IdentityLayer.INSTANCE;
      NetModel netModel = new NetModel(NetChain.join( //
          new TakeListLayer(1, 1, 4), //
          FeatureEmbeddingLayer.of(fs, fi, fn), //
          new Flatten1Layer(), //
          optLinearChain.create(vs + vi + 4, 1), //
          SingletonLayer.VectorToScalar), SquaredErrorLoss.INSTANCE);
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
      Tensor y = ypred.subtract(ydata);
      Tensor eperc = Unitless.PERCENT.unitConvert().apply(Mean.ofVector(y.maps(Abs.FUNCTION))).maps(Round._1);
      Show show = new Show("error " + eperc);
      show.add(ListDiscretePlot.of(x, y));
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
