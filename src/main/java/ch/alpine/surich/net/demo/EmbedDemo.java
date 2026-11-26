// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ArrayPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.ArrayReshape;
import ch.alpine.tensor.alg.OrderedQ;
import ch.alpine.tensor.alg.PadRight;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Tuples;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.sca.Ceiling;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.pow.Sqrt;

@ReflectionMarker
class EmbedDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new EmbedComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new EmbedDemo().runStandalone();
  }
}

@ReflectionMarker
class EmbedComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer S = 3;
  public Integer V = 4;
  public Integer D = 8;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(Clips.absolute(2), Clips.absolute(2));
    final TrainingData trainingData;

    public InnerComponent() {
      super(0);
      Tensor xdata = Tuples.of(Range.of(0, V), S);
      Tensor ydata = Tensor.of(xdata.stream().map(OrderedQ::of).map(Boole::of));
      // assignment of random labels to points
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer inner = NetChains.linear(between, S * D, 16, 8, 1);
      NetModel netModel = NetModels.logits( //
          NetChains.embedding(V, D, inner), //
          ScalarLayer.VectorToScalar);
      NetLayer netLayer = netModel.netLayer();
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Tensor error = netLayer.forwardBatch(trainingData.xdata()).subtract(trainingData.ydata());
      int N = error.length();
      int ceil = Ceiling.intValueExact(Sqrt.FUNCTION.apply(RealScalar.of(N)));
      error = PadRight.zeros(ceil * ceil).apply(error);
      Tensor matrix = ArrayReshape.of(error, ceil, ceil);
      Show show = new Show("Error");
      show.add(ArrayPlot.of(matrix));
      show.render_autoIndent(graphics, list.get(0));
    }
  }
}
