// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ArrayPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.run.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ArrayReshape;
import ch.alpine.tensor.alg.Flatten;
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
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public InnerComponent() {
      Tensor xdata = Tuples.of(Range.of(0, V), S);
      Tensor ydata = Tensor.of(xdata.stream().map(OrderedQ::of).map(Boole::of).map(Tensors::of));
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
      NetLayer netLayer = NetChains.binary(NetChains.embedding(V, D, inner));
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain(netLayer, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
      NetLayer copy = InnerComponent.this.copy;
      if (Objects.nonNull(copy)) {
        List<Rectangle> list = ShowGridComponent.rectangles(3, getSize());
        {
          Tensor error = Flatten.of(copy.forwardBatch(trainingData.xdata()).subtract(trainingData.ydata()));
          int N = error.length();
          int ceil = Ceiling.intValueExact(Sqrt.FUNCTION.apply(RealScalar.of(N)));
          error = PadRight.zeros(ceil * ceil).apply(error);
          Tensor matrix = ArrayReshape.of(error, ceil, ceil);
          Show show = new Show("Error");
          show.add(ArrayPlot.of(matrix));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
          NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(3));
        }
      }
    }
  }
}
