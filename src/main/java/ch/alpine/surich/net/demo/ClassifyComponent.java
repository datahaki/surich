// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.opt.nd.BoxRandomSample;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomSample;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class ClassifyComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 30;
  public Integer labels = 3;
  public Integer depth = 2;
  public Integer hidden = 7;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;
  public ColorDataGradients cdg = ColorDataGradients.COPPER;

  @Override
  public NetTrainComponent get() {
    return new Interp1TrainComponent();
  }

  class Interp1TrainComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(Clips.absolute(2), Clips.absolute(2));
    final TrainingData trainingData;
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public Interp1TrainComponent() {
      Random random = new Random(seed);
      // assignment of random labels to points
      trainingData = new TrainingData( //
          RandomSample.of(new BoxRandomSample(cbb), random, samples), //
          RandomVariate.of(DiscreteUniformDistribution.forArray(labels), random, samples, 1));
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      int[] dims = new int[depth + 2];
      dims[0] = 2;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      dims[dims.length - 1] = labels;
      NetLayer netLayer = NetChains.argMax(NetChains.linear(between, dims));
      NetInitialize.of(netLayer);
      netTrain = new NetTrain(netLayer, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
      NetLayer copy = Interp1TrainComponent.this.copy;
      if (Objects.nonNull(copy)) {
        List<Rectangle> list = ShowGridComponent.rectangles(3, getSize());
        {
          Show show = new Show();
          show.add(DensityPlot.of((x, y) -> copy.forward(Tensors.of(x, y)).Get(0), cbb, cdg));
          Tensor xdata = trainingData.xdata();
          Tensor ydata = trainingData.ydata();
          for (int k = 0; k < labels; ++k) {
            Scalar label = RealScalar.of(k);
            Tensor p0 = Tensor.of(IntStream.range(0, xdata.length()) //
                .filter(i -> ydata.Get(i, 0).equals(label)).mapToObj(xdata::get));
            show.add(ListPlot.of(p0));
          }
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
          NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(3));
        }
      }
    };
  }
}
