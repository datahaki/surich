// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.PolygonPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.AbsSquared;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.tri.Sin;

@ReflectionMarker
class SwiGLUInterp2Demo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new SwiGLUInterp2Component();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new SwiGLUInterp2Demo().runStandalone();
  }
}

@ReflectionMarker
class SwiGLUInterp2Component implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absolute(1);
  public Integer seed = 0;
  public Integer samples = 100;
  public Integer hidden = 5;
  public Scalar l2 = RealScalar.of(0.001);
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 20;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  private static Scalar truth(Tensor x) {
    return Sin.FUNCTION.apply(x.Get(0)).multiply(x.Get(1));
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      Distribution dist = UniformDistribution.of(DOMAIN);
      Tensor X = RandomVariate.of(dist, randomGenerator, samples, 2);
      Tensor Y = Slash.of(X, x -> truth(x));
      trainingData = new TrainingData(X, Y);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      SwiGLULayer swiGLULayer = new SwiGLULayer(2, hidden, 1);
      swiGLULayer.netRegularization = L2Regularization.of(l2);
      NetModel netModel = new NetModel(NetChain.join( //
          swiGLULayer, //
          ScalarLayer.VectorToScalar), //
          SquaredErrorLoss.INSTANCE);
      NetLayer netLayer = netModel.netLayer();
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
      Clip clip = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
      CoordinateBoundingBox ext = CoordinateBoundingBox.of(clip, clip);
      ColorDataGradients cdg = ColorDataGradients.COPPER;
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(2);
      {
        Show show = new Show("function");
        ScalarBinaryOperator sbo = (x, y) -> truth(Tensors.of(x, y));
        show.add(DensityPlot.of(sbo, ext, cdg));
        show.add(PolygonPlot.of(cbb));
        show.add(ListPlot.of(trainingData.xdata()));
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        Show show = new Show("squared residual");
        ScalarBinaryOperator sbo = (x, y) -> {
          Tensor p = Unprotect.using(List.of(x, y));
          return (Scalar) netLayer.forward(p).subtract(truth(p)).maps(AbsSquared.FUNCTION);
        };
        show.add(DensityPlot.of(sbo, ext, cdg));
        // show.add(PolygonPlot.of(cbb));
        // show.add(ListPlot.of(trainingData.xdata()));
        show.render_autoIndent(graphics, list.get(1));
      }
    };
  }
}
