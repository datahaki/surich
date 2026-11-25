// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptLinearChain;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class SwiGLUClassifyDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new SwiGLUClassifyComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new SwiGLUClassifyDemo().runStandalone();
  }
}

@ReflectionMarker
class SwiGLUClassifyComponent implements Supplier<NetTrainComponent> {
  public static final Clip DOMAIN = Clips.absolute(2);
  public Integer seed = 0;
  public Integer samples = 30;
  public final OptLinearChain optLinear = new OptLinearChain();
  public final OptParam optParam = new OptParam();
  public Integer batch_size = 6;

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  static boolean labelFunc(int label) {
    return label % 2 == 0;
  }

  class InnerComponent extends NetTrainComponent {
    final CoordinateBoundingBox cbb = CoordinateBoundingBox.of(DOMAIN, DOMAIN);
    final TrainingData trainingData;
    final Tensor X1 = Tensors.reserve(samples / 2);
    final Tensor X2 = Tensors.reserve(samples / 2);

    public InnerComponent() {
      super(seed);
      // assignment of random labels to points
      Distribution radiusClass0 = UniformDistribution.of(0.0, 0.7);
      Distribution radiusClass1 = UniformDistribution.of(1.0, 1.7);
      Distribution angleDist = UniformDistribution.of(0.0, 2.0 * Math.PI);
      Tensor Y = Tensors.reserve(samples);
      for (int i = 0; i < samples; ++i) {
        boolean label = labelFunc(i);
        double r = RandomVariate.of(label ? radiusClass0 : radiusClass1, randomGenerator).number().doubleValue();
        double theta = RandomVariate.of(angleDist, randomGenerator).number().doubleValue();
        double x0 = r * Math.cos(theta);
        double x1 = r * Math.sin(theta);
        if (label) {
          X1.append(Tensors.vector(x0, x1));
        } else {
          X2.append(Tensors.vector(x0, x1));
        }
        // Two-output target vector: [1, 0] for class 0, [0, 1] for class 1
        Y.append(label ? Tensors.vector(1.0, 0.0) : Tensors.vector(0.0, 1.0));
        // Y.append(Boole.of(label == 0));
      }
      trainingData = new TrainingData(Join.of(X1, X2), Y);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      NetModel netModel2 = new NetModel(optLinear.swiGLU(2, 2), SquaredErrorLoss.INSTANCE);
      NetModel netModel = netModel2;
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer, randomGenerator);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(2);
      {
        Show show = new Show("training samples");
        show.add(ListPlot.of(X1));
        show.add(ListPlot.of(X2));
        show.setAspectRatioOne();
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        Show show = new Show("net foward of training samples");
        show.add(ListPlot.of(Slash.of(X1, netLayer::forward)));
        show.add(ListPlot.of(Slash.of(X2, netLayer::forward)));
        show.setAspectRatioOne();
        show.render_autoIndent(graphics, list.get(1));
      }
    };
  }
}
