// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer;
import ch.alpine.subare.net.lay.FeatureEmbeddingLayer.FeatureSlice;
import ch.alpine.subare.net.lay.IdentityLayer;
import ch.alpine.subare.net.lay.OneHotLayer;
import ch.alpine.subare.net.lay.ScalarEmbeddingLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Join;
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
  public final OptLinearChain optLinear = new OptLinearChain();
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
    final TrainingData trainingData;
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      Tensor X = RandomVariate.of(UniformDistribution.of(DOMAIN), random, samples, 1);
      Tensor I = RandomVariate.of(DiscreteUniformDistribution.forArray(V), random, samples, 1);
      Tensor xdata = Join.of(1, X, I);
      Tensor ydata = Slash.of(xdata, //
          xi -> featureFunc.create(xi.Get(1).number().intValue()).apply(xi.Get(0)));
      // assignment of random labels to points
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer embed = FeatureEmbeddingLayer.of( //
          new FeatureSlice(1, 1, IdentityLayer.INSTANCE), //
          oneHot //
              ? new FeatureSlice(1, V, new OneHotLayer(V))
              : new FeatureSlice(1, D, new ScalarEmbeddingLayer(V, D)) //
      );
      NetLayer netLayer = NetChains.scalar(NetChain.join(embed, optLinear.create(d_emb(), 1)));
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain(NetModels.squaredErrorLoss(netLayer), optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetLayer copy) {
      List<Rectangle> list = rectangles(4);
      {
        Clip clip = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
        Show show = new Show("approximation");
        show.add(ListPlot.of( //
            trainingData.xdata().get(Tensor.ALL, 0), //
            trainingData.ydata())).setAlpha(64);
        for (int i = 0; i < V; ++i) {
          Scalar si = RealScalar.of(i);
          Showable showable = show.add(Plot.of(x -> (Scalar) copy.forward(Tensors.of(x, si)), clip));
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
