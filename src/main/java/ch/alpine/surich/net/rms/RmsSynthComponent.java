// code by jph
package ch.alpine.surich.net.rms;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.DiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.RMSLayer;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class RmsSynthComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 30;
  public Tensor gamma = Tensors.vector(0.7, 0.3, 0.25);
  public Scalar noise_sigma = RealScalar.of(0.0001);
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  record RMSLinear(RMSLayer rmsLayer, LinearLayer linearLayer) {
    static RMSLinear create() {
      return new RMSLinear(new RMSLayer(), new LinearLayer(3, 1));
    }

    NetLayer consolidate() {
      linearLayer.netRegularization = NoRegularization.INSTANCE;
      return NetChains.binary(NetChain.join(rmsLayer, linearLayer));
    }
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      RMSLinear rmsLinear = RMSLinear.create();
      rmsLinear.rmsLayer.gamma = gamma;
      NetLayer netLayer = rmsLinear.consolidate();
      NetInitialize.of(netLayer);
      Tensor xdata = RandomVariate.of(DiscreteUniformDistribution.of(-3, 10), random, samples, 3);
      Tensor noise = RandomVariate.of(NormalDistribution.of(noise_sigma.zero(), noise_sigma), random, samples, 1);
      Tensor ydata = netLayer.forwardBatch(xdata).add(noise);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      RMSLinear rmsLinear = RMSLinear.create();
      NetLayer netLayer = rmsLinear.consolidate();
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
          Show show = new Show("error");
          Tensor Y = copy.forwardBatch(trainingData.xdata());
          Tensor d = Y.subtract(trainingData.ydata());
          show.add(DiscretePlot.of(x -> d.Get(x.number().intValue(), 0), Clips.positive(samples - 1)));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        }
      }
    };
  }
}
