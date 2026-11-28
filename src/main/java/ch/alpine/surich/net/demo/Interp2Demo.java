// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.List;

import ch.alpine.bridge.fig.Rasterize;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.PolygonPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.FourierLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.ext.Cache;
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
class Interp2Demo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new Interp2Component();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new Interp2Demo().runStandalone();
  }
}

@ReflectionMarker
class Interp2Component extends NetTrainComponentSupplier {
  private static final Clip DOMAIN = Clips.absoluteOne();
  public Integer variability = 5;
  public Integer samples = 50;
  public Integer numFreq = 40;
  public Scalar sigma = RealScalar.of(10);
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public Integer plotPoints = 20;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TensorUnaryOperator truth;
    final TrainingData trainingData;
    private final Cache<Dimension, BufferedImage> cache = Cache.of(this::compute, 1);

    BufferedImage compute(Dimension dimension) {
      CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
      Clip clip = Clips.operator(RealScalar.of(1.05)::multiply).apply(DOMAIN);
      CoordinateBoundingBox ext = CoordinateBoundingBox.of(clip, clip);
      ColorDataGradients cdg = ColorDataGradients.COPPER;
      Show show = new Show("function");
      ScalarBinaryOperator sbo = (x, y) -> (Scalar) truth.apply(Tensors.of(x, y));
      DensityPlot densityPlot = DensityPlot.of(sbo, ext, cdg);
      densityPlot.setPlotPoints(plotPoints);
      show.add(densityPlot);
      show.add(ListPlot.of(trainingData.xdata()));
      show.add(PolygonPlot.of(cbb));
      return new Rasterize(show, dimension).image();
    }

    public InnerComponent() {
      super(seed);
      Tensor disY = RandomVariate.of(NormalDistribution.standard(), randomGenerator, variability, variability);
      Interpolation interpolation = BSplineInterpolation.of(3, disY);
      ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(variability - 1));
      Tensor xdata = RandomVariate.of(UniformDistribution.of(DOMAIN), randomGenerator, samples, 2);
      truth = xy -> interpolation.get(xy.maps(s -> lin.apply(DOMAIN.rescale(s))));
      Tensor ydata = Slash.of(xdata, truth);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      FourierLayer fourierLayer = new FourierLayer(2, numFreq, sigma);
      NetLayer netLayer = NetChain.join(fourierLayer, optLinearChain.create(2 * numFreq, 1), SingletonLayer.VectorToScalar);
      return new NetModelData(new NetModel(netLayer, SquaredErrorLoss.INSTANCE), trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
      Clip clip = Clips.operator(RealScalar.of(1.05)::multiply).apply(DOMAIN);
      CoordinateBoundingBox ext = CoordinateBoundingBox.of(clip, clip);
      List<Rectangle> list = rectangles(2);
      ColorDataGradients cdg = ColorDataGradients.COPPER;
      {
        Rectangle rectangle = list.get(0);
        BufferedImage bufferedImage = cache.apply(rectangle.getSize());
        graphics.drawImage(bufferedImage, rectangle.x, rectangle.y, null);
      }
      {
        Show show = new Show("error");
        ScalarBinaryOperator sbo = (x, y) -> {
          Tensor p = Unprotect.using(List.of(x, y));
          return (Scalar) netLayer.forward(p).subtract(truth.apply(p));
        };
        DensityPlot densityPlot = DensityPlot.of(sbo, ext, cdg);
        densityPlot.setPlotPoints(plotPoints);
        show.add(densityPlot);
        show.add(ListPlot.of(trainingData.xdata()));
        show.add(PolygonPlot.of(cbb));
        show.render_autoIndent(graphics, list.get(1));
      }
    };
  }
}
