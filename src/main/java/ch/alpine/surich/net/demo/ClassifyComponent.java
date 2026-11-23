// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.PolygonPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptLinearChain;
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
  public final OptLinearChain optLinear = new OptLinearChain();
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;

  @Override
  public NetTrainComponent get() {
    return new Interp1TrainComponent();
  }

  class Interp1TrainComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;
    NetTrain netTrain;

    public Interp1TrainComponent() {
      Random random = new Random(seed);
      // assignment of random labels to points
      trainingData = new TrainingData( //
          RandomSample.of(new BoxRandomSample(cbb), random, samples), //
          RandomVariate.of(DiscreteUniformDistribution.forArray(labels), random, samples));
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetModel netModel = NetModels.argMax(optLinear.create(2, 3));
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(4);
      {
        Clip eval = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
        CoordinateBoundingBox ext = CoordinateBoundingBox.of(eval, eval);
        Show show = new Show();
        ColorDataGradients cdg = ColorDataGradients.COPPER;
        show.add(DensityPlot.of((x, y) -> (Scalar) netLayer.forward(Tensors.of(x, y)), ext, cdg));
        Tensor xdata = trainingData.xdata();
        Tensor ydata = trainingData.ydata();
        for (int k = 0; k < labels; ++k) {
          Scalar label = RealScalar.of(k);
          Tensor p0 = Tensor.of(IntStream.range(0, xdata.length()) //
              .filter(i -> ydata.Get(i).equals(label)).mapToObj(xdata::get));
          show.add(ListPlot.of(p0));
        }
        show.add(PolygonPlot.of(cbb)).setAlpha(64);
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        NetShows.pMatrix(netLayer).render_autoIndent(graphics, list.get(3));
      }
    };
  }
}
