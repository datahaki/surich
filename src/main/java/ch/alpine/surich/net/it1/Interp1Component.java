// code by jph
package ch.alpine.surich.net.it1;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.NetWrap;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.tensor.RealScalar;
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
  public NetOptimizers GD = NetOptimizers.MOMENT;
  public Scalar eta = RealScalar.of(0.01);
  public Scalar beta1 = RealScalar.of(0.9);
  public Scalar beta2 = RealScalar.of(0.99);
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
    public void runAsBatches() {
      int[] dims = new int[depth + 2];
      dims[0] = 1;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      dims[dims.length - 1] = 1;
      NetLayer netLayer = NetChains.linear(between.supplier(), dims);
      NetInitialize.of(netLayer);
      netTrain = new NetTrain(netLayer, GD.create(netLayer, eta, beta1, beta2));
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
          Show show = new Show("interpolation");
          show.add(Plot.of(truth, DOMAIN));
          show.add(Plot.of(NetWrap.asSUO11(copy), DOMAIN)).setLabel("gelu");
          Tensor xdata = trainingData.xdata();
          Tensor ydata = trainingData.ydata();
          show.add(ListPlot.of(Join.of(1, xdata, ydata)));
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
          Show show = new Show("losses");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
          // show1.add(ListLinePlot.of(netLayerWrap1.netTrain.tloss.getTable())).setLabel("tanh");
          show.render_autoIndent(graphics, list.get(2));
        }
      }
    };
  }
}
