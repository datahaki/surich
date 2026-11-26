// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelDataBatch;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.exp.Exp;
import ch.alpine.tensor.sca.pow.Power;
import ch.alpine.tensor.sca.tri.Cos;
import ch.alpine.tensor.sca.tri.Sin;

@ReflectionMarker
class SwiGLUInterp1Demo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new SwiGLUInterp1Component();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new SwiGLUInterp1Demo().runStandalone();
  }
}

@ReflectionMarker
class SwiGLUInterp1Component extends NetTrainComponentSupplier {
  enum SwiGluFunc implements ScalarUnaryOperator {
    X2 {
      @Override
      public Scalar apply(Scalar x) {
        return x.multiply(x);
      }
    },
    X3 {
      @Override
      public Scalar apply(Scalar x) {
        return Times.of(x, x, x);
      }
    },
    X4 {
      @Override
      public Scalar apply(Scalar x) {
        return Power.of(x, 4);
      }
    },
    X2_X4 {
      @Override
      public Scalar apply(Scalar x) {
        return x.multiply(x).add(Power.of(x, 4));
      }
    },
    X_SIN {
      @Override
      public Scalar apply(Scalar x) {
        return Sin.FUNCTION.apply(x).multiply(x);
      }
    },
    X2_COS {
      @Override
      public Scalar apply(Scalar x) {
        return Cos.FUNCTION.apply(x).multiply(x).multiply(x);
      }
    },
    X2_EXP {
      @Override
      public Scalar apply(Scalar x) {
        return Exp.FUNCTION.apply(x.multiply(x).negate()).multiply(x).multiply(x);
      }
    },
  }

  public Scalar domain = RealScalar.ONE;
  public Integer samples = 100;
  public Integer hidden = 10;
  public Scalar l2 = RealScalar.ZERO;
  public SwiGluFunc swiGluFunc = SwiGluFunc.X2_COS;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      Clip clip = Clips.absolute(domain);
      Distribution dist = UniformDistribution.of(clip);
      Tensor X = RandomVariate.of(dist, randomGenerator, samples);
      Tensor Y = X.maps(swiGluFunc);
      trainingData = new TrainingData(X, Y);
    }

    @Override
    public NetModelDataBatch createNetModelDataBatch() {
      SwiGLULayer swiGLULayer = new SwiGLULayer(1, hidden, 1);
      swiGLULayer.netRegularization = L2Regularization.of(l2);
      NetModel netModel = new NetModel(NetChain.join( //
          ScalarLayer.ScalarToVector, //
          swiGLULayer, //
          ScalarLayer.VectorToScalar), //
          SquaredErrorLoss.INSTANCE);
      return new NetModelDataBatch(netModel, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      Clip clip = Clips.absolute(domain);
      Clip ext = Clips.operator(RealScalar.of(1.2)::multiply).apply(clip);
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Show show = new Show("function");
      show.add(Plot.of(swiGluFunc, ext)).setLabel("truth");
      show.add(Plot.of(x -> (Scalar) netLayer.forward(x), ext)).setLabel("net.forward");
      show.add(ListPlot.of(trainingData.xdata(), trainingData.ydata())).setLabel("training data");
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
