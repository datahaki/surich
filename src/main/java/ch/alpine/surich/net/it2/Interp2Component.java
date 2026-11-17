// code by jph
package ch.alpine.surich.net.it2;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetParamStatTracker;
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
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.itp.BSplineInterpolation;
import ch.alpine.tensor.itp.Interpolation;
import ch.alpine.tensor.itp.LinearInterpolation;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Interp2Component implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer variability = 5;
  public Integer samples = 50;
  public Integer depth = 2;
  public Integer hidden = 7;
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public NetOptimizers GD = NetOptimizers.MOMENT;
  public Scalar eta = RealScalar.of(0.01);
  public Scalar beta1 = RealScalar.of(0.9);
  public Scalar beta2 = RealScalar.of(0.99);
  public Integer batch_size = 10;
  public ColorDataGradients cdg = ColorDataGradients.COPPER;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TensorUnaryOperator truth;
    final TrainingData trainingData;
    final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
    NetTrain netTrain;

    public InnerComponent() {
      Random random = new Random(seed);
      Tensor disY = RandomVariate.of(NormalDistribution.standard(), random, variability, variability);
      Interpolation interpolation = BSplineInterpolation.of(3, disY);
      ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(variability - 1));
      Tensor xdata = RandomVariate.of(UniformDistribution.of(DOMAIN), random, samples, 2);
      // FIXME without Tensors.of(...) this should crash but doesnt !!!
      truth = xy -> Tensors.of(interpolation.get(xy.maps(s -> lin.apply(DOMAIN.rescale(s)))));
      Tensor ydata = truth.slash(xdata);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public void runAsBatches() {
      int[] dims = new int[depth + 2];
      dims[0] = 2;
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
        List<Rectangle> list = ShowGridComponent.rectangles(4, getSize());
        {
          Show show = new Show("function");
          ScalarBinaryOperator sbo = (x, y) -> truth.apply(Tensors.of(x, y)).Get(0);
          CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
          show.add(DensityPlot.of(sbo, cbb, cdg));
          show.add(ListPlot.of(trainingData.xdata()));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          Show show = new Show("interpolation error");
          ScalarBinaryOperator sbo = (x, y) -> {
            Tensor p = Unprotect.using(List.of(x, y));
            return copy.forward(p).subtract(truth.apply(p)).Get(0);
          };
          CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
          show.add(DensityPlot.of(sbo, cbb, cdg));
          show.add(ListPlot.of(trainingData.xdata()));
          show.render_autoIndent(graphics, list.get(1));
        }
        {
          Show show = new Show("parameter stats");
          Tensor[] tensors = netParamStatTracker.tableBuilder().oneToMany();
          for (int i = 0; i < tensors.length; ++i)
            show.add(ListLinePlot.of(tensors[i])).setLabel(NetParamStatTracker.LABELS[i]);
          show.render_autoIndent(graphics, list.get(2));
        }
        {
          Show show = new Show("losses");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
          show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
          // show1.add(ListLinePlot.of(netLayerWrap1.netTrain.tloss.getTable())).setLabel("tanh");
          show.render_autoIndent(graphics, list.get(3));
        }
      }
    };
  }
}
