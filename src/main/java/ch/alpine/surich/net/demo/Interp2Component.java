// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Supplier;

import ch.alpine.bridge.awt.RenderQuality;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.PolygonPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.run.NetParamStatTracker;
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
  private static final Clip DOMAIN = Clips.absoluteOne();
  private static final ColorDataGradients CDG = ColorDataGradients.COPPER;
  public Integer seed = 0;
  public Integer variability = 5;
  public Integer samples = 50;
  public final OptLinearChain optLinear = new OptLinearChain();
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;

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
      truth = xy -> interpolation.get(xy.maps(s -> lin.apply(DOMAIN.rescale(s))));
      Tensor ydata = Slash.of(xdata, truth);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetLayer netLayer = NetChains.scalar(optLinear.create(2, 1));
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
        CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
        Clip clip = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
        CoordinateBoundingBox ext = CoordinateBoundingBox.of(clip, clip);
        {
          RenderQuality.setQuality(graphics);
          NetShows.layers(graphics, new Point(), copy);
        }
        List<Rectangle> list = rectangles(6);
        {
          Show show = new Show("function");
          ScalarBinaryOperator sbo = (x, y) -> (Scalar) truth.apply(Tensors.of(x, y));
          show.add(DensityPlot.of(sbo, ext, CDG));
          show.add(PolygonPlot.of(cbb));
          show.add(ListPlot.of(trainingData.xdata()));
          show.render_autoIndent(graphics, list.get(0));
        }
        {
          Show show = new Show("error");
          ScalarBinaryOperator sbo = (x, y) -> {
            Tensor p = Unprotect.using(List.of(x, y));
            return (Scalar) copy.forward(p).subtract(truth.apply(p));
          };
          show.add(DensityPlot.of(sbo, ext, CDG));
          show.add(PolygonPlot.of(cbb));
          show.add(ListPlot.of(trainingData.xdata()));
          show.render_autoIndent(graphics, list.get(1));
        }
        {
          NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(2));
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(3));
          NetShows.pMatrix(copy).render_autoIndent(graphics, list.get(4));
        }
      }
    };
  }
}
