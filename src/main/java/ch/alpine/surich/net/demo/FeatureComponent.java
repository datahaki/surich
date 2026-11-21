// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer.NetFeature;
import ch.alpine.subare.net.lay.IdentityLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.OneHotLayer;
import ch.alpine.subare.net.lay.ScalarEmbeddingLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class FeatureComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public FeatureFunc featureFunc = FeatureFunc.Chebyshev_T;
  public Integer samples = 500;
  public Integer V = 4;
  public Integer D = 3;
  public Boolean oneHot = false;
  public Integer depth = 3;
  public Integer hidden = 32;
  public ColorDataGradients cdg = ColorDataGradients.THERMOMETER;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 50;

  private int d_emb() {
    return 1 + (oneHot ? V : D);
  }

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
      Random random = new Random(seed);
      Tensor X = RandomVariate.of(UniformDistribution.of(-1, 1), random, samples, 1);
      Tensor I = RandomVariate.of(DiscreteUniformDistribution.forArray(V), random, samples, 1);
      Tensor xdata = Join.of(1, X, I);
      Tensor ydata = Tensor.of(xdata.stream() //
          .map(xi -> Tensors.of(featureFunc.create(xi.Get(1).number().intValue()).apply(xi.Get(0)))));
      // assignment of random labels to points
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      int[] dims = new int[depth + 2];
      dims[0] = d_emb();
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      dims[dims.length - 1] = 1;
      NetLayer inner = NetChains.linear(between, dims);
      NetLayer embed = FeatureEmbeddingLayer.of( //
          new NetFeature(1, 1, IdentityLayer.INSTANCE), //
          oneHot //
              ? new NetFeature(1, V, new OneHotLayer(V))
              : new NetFeature(1, D, new ScalarEmbeddingLayer(V, D)) //
      );
      NetLayer netLayer = NetChain.join(embed, inner);
      NetInitialize.of(netLayer);
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
          Show show = new Show();
          show.add(ListPlot.of( //
              trainingData.xdata().get(Tensor.ALL, 0), //
              trainingData.ydata().get(Tensor.ALL, 0))).setAlpha(64);
          for (int i = 0; i < V; ++i) {
            Scalar si = RealScalar.of(i);
            Showable showable = show.add(Plot.of(x -> copy.forward(Tensors.of(x, si)).Get(0), Clips.absoluteOne()));
            showable.setLabel(featureFunc + "_" + i);
          }
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(1));
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(2));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(3));
        }
      }
    }
  }
}
