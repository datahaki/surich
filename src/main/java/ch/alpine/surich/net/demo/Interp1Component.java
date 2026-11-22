// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.GraphicsGrid;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
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
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.api.ScalarTensorFunction;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.itp.BSplineFunctionString;
import ch.alpine.tensor.itp.LinearInterpolation;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Interp1Component implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer variability = 5;
  public Integer samples = 30;
  public Integer depth = 2;
  public Integer hidden = 7;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final ScalarUnaryOperator truth;
    final TrainingData trainingData;
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      Tensor disY = RandomVariate.of(NormalDistribution.standard(), random, variability);
      ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
      ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(variability - 1));
      Tensor xdata = RandomVariate.of(UniformDistribution.of(DOMAIN), random, samples, 1);
      truth = s -> (Scalar) stf.apply(lin.apply(DOMAIN.rescale(s)));
      Tensor ydata = xdata.maps(truth);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      int[] dims = new int[depth + 2];
      dims[0] = 1;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      dims[dims.length - 1] = 1;
      NetLayer netLayer = NetChains.linear(between, dims);
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
        List<Rectangle> list = GraphicsGrid.rectangles(3, getSize());
        {
          Show show = new Show("interpolation");
          show.add(Plot.of(truth, DOMAIN));
          show.add(Plot.of(NetLayers.asSUO11(copy), DOMAIN)).setLabel("gelu");
          Tensor xdata = trainingData.xdata();
          Tensor ydata = trainingData.ydata();
          show.add(ListPlot.of(Join.of(1, xdata, ydata)));
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
