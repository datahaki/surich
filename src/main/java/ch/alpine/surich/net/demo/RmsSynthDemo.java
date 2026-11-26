// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListDiscretePlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.lay.RMSLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainComponentSupplier;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.DiscreteUniformDistribution;

@ReflectionMarker
class RmsSynthDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new RmsSynthComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new RmsSynthDemo().runStandalone();
  }
}

@ReflectionMarker
class RmsSynthComponent extends NetTrainComponentSupplier {
  public Integer samples = 30;
  public Scalar noise_sigma = RealScalar.of(0.0001);

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  record RMSLinear(RMSLayer rmsLayer, LinearLayer linearLayer) {
    static RMSLinear create() {
      return new RMSLinear(new RMSLayer(), new LinearLayer(3, 1));
    }

    NetModel consolidate() {
      linearLayer.netRegularization = NoRegularization.INSTANCE;
      return NetModels.sigmoid( //
          rmsLayer, //
          linearLayer, //
          SingletonLayer.VectorToScalar);
    }
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;

    public InnerComponent() {
      super(seed);
      RMSLinear rmsLinear = RMSLinear.create();
      NetLayer netLayer = rmsLinear.consolidate().netLayer();
      NetLayers.initialize(netLayer, randomGenerator);
      Tensor xdata = RandomVariate.of(DiscreteUniformDistribution.of(-3, 10), randomGenerator, samples, 3);
      Tensor noise = RandomVariate.of(NormalDistribution.of(noise_sigma.zero(), noise_sigma), randomGenerator, samples);
      Tensor ydata = netLayer.forwardBatch(xdata).add(noise);
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      RMSLinear rmsLinear = RMSLinear.create();
      NetModel netModel = rmsLinear.consolidate();
      return new NetModelData(netModel, trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(1);
      Show show = new Show("error");
      Tensor Y = netLayer.forwardBatch(trainingData.xdata());
      Tensor d = Y.subtract(trainingData.ydata());
      show.add(ListDiscretePlot.of(Range.of(0, d.length()), d));
      show.render_autoIndent(graphics, list.get(0));
    };
  }
}
