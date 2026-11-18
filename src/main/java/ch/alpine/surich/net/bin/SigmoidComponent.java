// code by jph
package ch.alpine.surich.net.bin;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.grd.AxisMap;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.col.ColorDataGradients;
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
  public Integer samples = 30;
  public Integer depth = 2;
  public Integer hidden = 7;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;
  public ColorDataGradients cdg = ColorDataGradients.COPPER;

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
      // assignment of random labels to points
      trainingData = new TrainingData( //
          RandomSample.of(new BoxRandomSample(cbb), random, samples), //
          RandomVariate.of(BernoulliDistribution.of(0.5), random, samples, 1));
    }

    @Override
    public void runAsBatches() {
      int[] dims = new int[depth + 2];
      dims[0] = 2;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      dims[dims.length - 1] = 1;
      NetLayer netLayer = NetChains.binary(NetChains.linear(between.supplier(), dims));
      NetInitialize.of(netLayer);
      netTrain = new NetTrain(netLayer, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      netTrain.runAsBatches(trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics) {
      NetLayer copy = InnerComponent.this.copy;
      if (Objects.nonNull(copy)) {
        List<Rectangle> list = ShowGridComponent.rectangles(3, getSize());
        {
          Show show = new Show();
          ScalarBinaryOperator sbo = (x, y) -> copy.forward(Tensors.of(x, y)).Get(0);
          show.add(DensityPlot.of(sbo, cbb, cdg));
          Tensor xdata = trainingData.xdata();
          Tensor ydata = trainingData.ydata();
          Tensor p0 = Tensor.of(IntStream.range(0, xdata.length()).filter(i -> Scalars.isZero(ydata.Get(i, 0))).mapToObj(xdata::get));
          Tensor p1 = Tensor.of(IntStream.range(0, xdata.length()).filter(i -> !Scalars.isZero(ydata.Get(i, 0))).mapToObj(xdata::get));
          show.add(ListPlot.of(p0));
          show.add(ListPlot.of(p1));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          Show show = new Show("parameter stats");
          Tensor[] tensors = netParamStatTracker.tableBuilder().oneToMany();
          for (int i = 0; i < tensors.length; ++i)
            show.add(ListLinePlot.of(tensors[i])).setLabel(NetParamStatTracker.LABELS[i]);
          show.render_autoIndent(graphics, list.get(1));
        }
        {
          Show show = new Show();
          show.setShowLabel("losses");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
          show.showOptions.axisOptionsY.axisMap = AxisMap.LOG;
          show.render_autoIndent(graphics, list.get(2));
        }
      }
    };
  }
}
