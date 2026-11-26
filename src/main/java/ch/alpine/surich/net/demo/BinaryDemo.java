// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.stream.IntStream;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.ForeAndBackPalettes;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptColor;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.api.TensorScalarFunction;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.opt.nd.BoxRandomSample;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomSample;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.tri.Sin;

@ReflectionMarker
class BinaryDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new BinaryComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new BinaryDemo().runStandalone();
  }
}

@ReflectionMarker
class BinaryComponent extends NetTrainComponentSupplier {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer samples = 30;
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public final OptColor optColor = new OptColor();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      // assignment of random labels to points
      Tensor X = RandomSample.of(new BoxRandomSample(cbb), randomGenerator, samples);
      TensorScalarFunction tsf = xy -> Boole.of(Scalars.lessEquals(xy.Get(1), //
          RealScalar.of(0.6).multiply(Sin.FUNCTION.apply(xy.Get(0).multiply(RealScalar.of(8)))) //
      ));
      trainingData = new TrainingData(X, Slash.of(X, tsf));
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      NetModel netModel = NetModels.logits( //
          optLinearChain.create(2, 1), //
          SingletonLayer.VectorToScalar);
      return new NetModelData(netModel, trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      ForeAndBackPalettes foreAndBackPalettes = optColor.create();
      Show show = new Show();
      ScalarBinaryOperator sbo = (x, y) -> (Scalar) netLayer.forward(Tensors.of(x, y));
      Clip clip = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
      CoordinateBoundingBox cbb = CoordinateBoundingBox.of(clip, clip);
      show.add(DensityPlot.of(sbo, cbb, foreAndBackPalettes.cdb()));
      Tensor xdata = trainingData.xdata();
      Tensor ydata = trainingData.ydata();
      for (int i = 0; i < 2; ++i) {
        Scalar label = RealScalar.of(i);
        Tensor points = Tensor.of(IntStream.range(0, xdata.length()) //
            .filter(j -> ydata.Get(j).equals(label)) //
            .mapToObj(xdata::get));
        show.add(ListPlot.of(points)).setColor(foreAndBackPalettes.cdi().getColor(i));
      }
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
