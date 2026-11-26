// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.ErrorType;
import ch.alpine.surich.net.NetModelDataBatch;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
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
class Interp1Demo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new Interp1Component();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new Interp1Demo().runStandalone();
  }
}

@ReflectionMarker
class Interp1Component extends NetTrainComponentSupplier {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer variability = 5;
  public Integer samples = 30;
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public ErrorType errorType = ErrorType._SquaredErrorLoss;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final ScalarUnaryOperator truth;
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      Tensor disY = RandomVariate.of(NormalDistribution.standard(), randomGenerator, variability);
      ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
      ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(variability - 1));
      Tensor xdata = RandomVariate.of(UniformDistribution.of(DOMAIN), randomGenerator, samples);
      truth = s -> (Scalar) stf.apply(lin.apply(DOMAIN.rescale(s)));
      Tensor ydata = xdata.maps(truth);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public NetModelDataBatch createNetModelDataBatch() {
      NetModel netModel = new NetModel(optLinearChain.asSUO(), errorType.lossFunction);
      return new NetModelDataBatch(netModel, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Clip eval = Clips.operator(RealScalar.of(1.5)::multiply).apply(DOMAIN);
      Show show = new Show("interpolation");
      show.add(Plot.of(truth, eval));
      show.add(Plot.of(s -> (Scalar) netLayer.forward(s), eval)).setLabel("gelu");
      Tensor xdata = trainingData.xdata();
      Tensor ydata = trainingData.ydata();
      show.add(ListPlot.of(xdata, ydata));
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
