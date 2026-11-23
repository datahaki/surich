// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.GraphicsGrid;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.RMSLayer;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;

@ReflectionMarker
class RmsSynthComponent implements Supplier<NetTrainComponent> {
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

    NetModel consolidate() {
      linearLayer.netRegularization = NoRegularization.INSTANCE;
      return NetModels.binary(NetChain.join(rmsLayer, linearLayer));
    }
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      RMSLinear rmsLinear = RMSLinear.create();
      rmsLinear.rmsLayer.gamma = gamma;
      NetModel netModel = rmsLinear.consolidate();
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer);
      Tensor xdata = RandomVariate.of(DiscreteUniformDistribution.of(-3, 10), random, samples, 3);
      Tensor noise = RandomVariate.of(NormalDistribution.of(noise_sigma.zero(), noise_sigma), random, samples);
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
      NetModel netModel = rmsLinear.consolidate();
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetLayer copy) {
      List<Rectangle> list = GraphicsGrid.rectangles(3, getSize());
      {
        Show show = new Show("error");
        Tensor Y = copy.forwardBatch(trainingData.xdata());
        Tensor d = Y.subtract(trainingData.ydata());
        show.add(ListDiscretePlot.of(Range.of(0, d.length()), d));
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(3));
      }
    };
  }
}
