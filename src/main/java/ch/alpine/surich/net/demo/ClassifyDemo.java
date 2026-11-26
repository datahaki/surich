// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.stream.IntStream;

import ch.alpine.bridge.col.HuePalette;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.DensityPlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.PolygonPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelDataBatch;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.col.ColorDataIndexed;
import ch.alpine.tensor.col.ColorDataLists;
import ch.alpine.tensor.opt.nd.BoxRandomSample;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.RandomSample;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class ClassifyDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new ClassifyComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new ClassifyDemo().runStandalone();
  }
}

@ReflectionMarker
class ClassifyComponent extends NetTrainComponentSupplier {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer samples = 30;
  public Integer labels = 3;
  public final OptLinearChain optLinearChain = new OptLinearChain();
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

    public InnerComponent() {
      super(seed);
      // assignment of random labels to points
      trainingData = new TrainingData( //
          RandomSample.of(new BoxRandomSample(cbb), randomGenerator, samples), //
          RandomVariate.of(DiscreteUniformDistribution.forArray(labels), randomGenerator, samples));
    }

    @Override
    public NetModelDataBatch createNetModelDataBatch() {
      NetModel netModel = NetModels.softmax_argMax(optLinearChain.create(2, labels));
      return new NetModelDataBatch(netModel, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      ColorDataIndexed cdi = cdl.cyclic();
      ColorDataIndexed cdb = HuePalette.of(cdi).create(sat, val);
      Clip eval = Clips.operator(RealScalar.of(1.2)::multiply).apply(DOMAIN);
      CoordinateBoundingBox ext = CoordinateBoundingBox.of(eval, eval);
      Show show = new Show();
      show.add(DensityPlot.of((x, y) -> (Scalar) netLayer.forward(Tensors.of(x, y)), ext, cdb));
      Tensor xdata = trainingData.xdata();
      Tensor ydata = trainingData.ydata();
      for (int k = 0; k < labels; ++k) {
        Scalar label = RealScalar.of(k);
        Tensor p0 = Tensor.of(IntStream.range(0, xdata.length()) //
            .filter(i -> ydata.Get(i).equals(label)).mapToObj(xdata::get));
        Showable showable = show.add(ListPlot.of(p0));
        showable.setColor(cdi.getColor(k));
      }
      show.add(PolygonPlot.of(cbb)).setAlpha(64);
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
