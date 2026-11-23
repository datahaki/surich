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
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.MatrixLayer;
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
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class MatrixSynthComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 200;
  public Integer dim0 = 3;
  public Integer dim1 = 5;
  public Integer dim2 = 2;
  public Scalar noise_sigma = RealScalar.of(0.0001);
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  private NetLayer create() {
    MatrixLayer n1 = new MatrixLayer(dim0, dim1);
    n1.netRegularization = NoRegularization.INSTANCE;
    MatrixLayer n2 = new MatrixLayer(dim1, dim2);
    n2.netRegularization = NoRegularization.INSTANCE;
    MatrixLayer n3 = new MatrixLayer(dim2, 1);
    n3.netRegularization = NoRegularization.INSTANCE;
    return NetChain.join(n1, n2, n3);
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      NetLayer netLayer = create();
      NetLayers.initialize(netLayer);
      Tensor xdata = RandomVariate.of(UniformDistribution.of(-1, 1), random, samples, dim0);
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
      NetLayer netLayer = create();
      NetLayers.initialize(netLayer);
      netTrain = new NetTrain( //
          NetModels.squaredErrorLoss(netLayer), //
          optParam.create(netLayer));
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
        show.add(ListDiscretePlot.of(Range.of(0, d.length()), Flatten.of(d)));
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
