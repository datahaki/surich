// code by jph
package ch.alpine.surich.net.it2;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.ElementwiseLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetMomentumGradientDescent;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.opt.nd.BoxRandomSample;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomSample;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class SigmoidComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer variability = 5;
  public Integer samples = 30;
  public Integer depth = 2;
  public Integer hidden = 7;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public Scalar beta = RealScalar.of(0.9);
  public Scalar eta = RealScalar.of(0.01);
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
      Tensor xdata = RandomSample.of(new BoxRandomSample(cbb), random, samples);
      // assignment of random labels to points
      // Tensor ydata = Tensor.of(xdata.stream() //
      // .map(SimplexContinuousNoise.FUNCTION) //
      // .map(UnitStep.FUNCTION) //
      // .map(Tensors::of));
      Tensor ydata = RandomVariate.of(BernoulliDistribution.of(0.5), random, samples, 1);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public void runAsBatches() {
      int[] dims = new int[depth + 2];
      dims[0] = 2;
      dims[dims.length - 1] = 1;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      NetLayer netLayer = NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), dims));
      NetInitialize.of(netLayer);
      NetOptimizer netOptimize = new NetMomentumGradientDescent(netLayer, beta, eta);
      netTrain = new NetTrain(netLayer, netOptimize);
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      netTrain.runAsBatches(trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
      Dimension dimension = getSize();
      NetLayer copy = Interp1TrainComponent.this.copy;
      if (Objects.nonNull(copy)) {
        {
          Show show = new Show();
          show.add(DensityPlot.of((x, y) -> copy.forward(Tensors.of(x, y)).Get(0), cbb, cdg));
          Tensor xdata = trainingData.xdata();
          Tensor ydata = trainingData.ydata();
          Tensor p0 = Tensor.of(IntStream.range(0, xdata.length()).filter(i -> Scalars.isZero(ydata.Get(i, 0))).mapToObj(xdata::get));
          Tensor p1 = Tensor.of(IntStream.range(0, xdata.length()).filter(i -> !Scalars.isZero(ydata.Get(i, 0))).mapToObj(xdata::get));
          show.add(ListPlot.of(p0));
          show.add(ListPlot.of(p1));
          show.render_autoIndent(graphics, new Rectangle(0, 0, dimension.width / 2, dimension.height / 2));
          //
          // Show show = new Show();
          // show.setShowLabel("interpolation");
          // show.add(Plot.of(NetWrap.asSUO11(copy), DOMAIN)).setLabel("gelu");
          // show.add(ListPlot.of(Join.of(1, xdata, ydata)));
          // show.render_autoIndent(graphics, new Rectangle(0, 0, dimension.width / 2, dimension.height / 2));
        }
        {
          Show show = new Show();
          show.setShowLabel("parameter stats");
          TableBuilder tableBuilder = netParamStatTracker.getTableCopy();
          for (int i = 1; i < NetParamStatTracker.LABELS.length; ++i) {
            Showable showable = show.add(ListLinePlot.of(tableBuilder.getColumns(0, i)));
            showable.setLabel(NetParamStatTracker.LABELS[i]);
          }
          show.render_autoIndent(graphics, new Rectangle(dimension.width / 2, 0, dimension.width / 2, dimension.height / 2));
        }
        {
          Show show = new Show();
          show.setShowLabel("losses");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
          // show1.add(ListLinePlot.of(netLayerWrap1.netTrain.tloss.getTable())).setLabel("tanh");
          show.render_autoIndent(graphics, new Rectangle(0, dimension.height / 2, dimension.width / 2, dimension.height / 2));
        }
      }
    };
  }
}
