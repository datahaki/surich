// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import ch.alpine.bridge.col.HuePalette;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarBinaryOperator;
import ch.alpine.tensor.api.TensorScalarFunction;
import ch.alpine.tensor.col.ColorDataIndexed;
import ch.alpine.tensor.col.ColorDataLists;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.opt.nd.BoxRandomSample;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomSample;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.UnitStep;
import ch.alpine.tensor.sca.tri.Sin;

@ReflectionMarker
class BinaryDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new BinaryComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new BinaryDemo().runStandalone();
  }
}

@ReflectionMarker
class BinaryComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer seed = 0;
  public Integer samples = 30;
  public final OptLinearChain optLinearChain = new OptLinearChain();
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 10;
  public ColorDataLists cdl = ColorDataLists._097;
  public Scalar sat = RealScalar.of(0.5);
  public Scalar val = RealScalar.of(0.5);

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;
    NetTrain netTrain;

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
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetModel netModel = NetModels.logits(optLinearChain.create(2, 1));
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer, randomGenerator);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(4);
      {
        ColorDataIndexed cdi = cdl.cyclic();
        ColorDataIndexed cdb = HuePalette.of(cdi).create(sat, val);
        Show show = new Show();
        ScalarBinaryOperator sbo = (x, y) -> (Scalar) netLayer.forward(Tensors.of(x, y)).maps(UnitStep.FUNCTION);
        Clip clip = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
        CoordinateBoundingBox cbb = CoordinateBoundingBox.of(clip, clip);
        show.add(DensityPlot.of(sbo, cbb, cdb));
        Tensor xdata = trainingData.xdata();
        Tensor ydata = trainingData.ydata();
        for (int i = 0; i < 2; ++i) {
          Scalar label = RealScalar.of(i);
          Tensor points = Tensor.of(IntStream.range(0, xdata.length()) //
              .filter(j -> ydata.Get(j).equals(label)) //
              .mapToObj(xdata::get));
          show.add(ListPlot.of(points)).setColor(cdi.getColor(i));
        }
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        NetShows.pMatrix(netLayer).render_autoIndent(graphics, list.get(3));
      }
    };
  }
}
