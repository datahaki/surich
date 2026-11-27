// code by jph
package ch.alpine.surich.net.img;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ImagePlot;
import ch.alpine.bridge.math.Meshgrid;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
import ch.alpine.subare.net.lay.TrainableFourierLayer;
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
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Reverse;
import ch.alpine.tensor.alg.Transpose;
import ch.alpine.tensor.io.ImageFormat;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.itp.Interpolation;
import ch.alpine.tensor.itp.LinearInterpolation;
import ch.alpine.tensor.opt.nd.CoordinateBoundingBox;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class ImageApproxDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final NetTrainComponentSupplier supplier = new ImageApproxComponent();

  @Override
  protected NetTrainComponentSupplier supplier() {
    return supplier;
  }

  static void main() {
    new ImageApproxDemo().runStandalone();
  }
}

@ReflectionMarker
class ImageApproxComponent extends NetTrainComponentSupplier {
  enum ImageName {
    gray_wa,
    gray_ma,
    gray_ec,
  }

  private static final Scalar xFF = RealScalar.of(255.0);
  public ImageName imageName = ImageName.gray_ma;
  public Integer samples = 20;
  public Integer numFreq = 10;
  public Scalar sigma = RealScalar.of(10);
  public final OptLinearChain optLinearChain = new OptLinearChain();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    final Tensor image;
    final Dimension dimension;
    final CoordinateBoundingBox networkDomain;
    final Meshgrid meshgrid;

    public InnerComponent() {
      super(seed);
      image = Import.of("ch/alpine/surich/net/img/" + imageName.name() + ".jpg");
      dimension = new Dimension(Unprotect.dimension1(image), image.length());
      Interpolation interpolation = LinearInterpolation.of(Transpose.of(Reverse.of(image.divide(xFF))));
      {
        Scalar redux = RealScalar.of((double) Math.max(dimension.width, dimension.height));
        // for example [0, 1] x [0, 0.75]
        networkDomain = CoordinateBoundingBox.of( //
            Clips.positive(RealScalar.of(dimension.width).divide(redux)), //
            Clips.positive(RealScalar.of(dimension.height).divide(redux)));
      }
      Scalar redux = RealScalar.of((double) Math.max(dimension.width - 1, dimension.height - 1));
      notes.add("network domain: " + networkDomain);
      meshgrid = Meshgrid.of(networkDomain, AwtUtil.aspect(samples, dimension));
      notes.add("mesh grid: " + meshgrid);
      Tensor px = meshgrid.px_incr();
      Tensor py = meshgrid.py_decr();
      Tensor xdata = Tensors.reserve(samples * samples);
      Tensor ydata = Tensors.reserve(samples * samples);
      for (Tensor x : px)
        for (Tensor y : py) {
          Tensor p = Unprotect.byRef(x, y);
          xdata.append(p);
          Tensor pixel = p.multiply(redux);
          ydata.append(interpolation.get(pixel));
        }
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      NetLayer enhanceLayer = new TrainableFourierLayer(2, numFreq, sigma);
      NetLayer netLayer = NetChain.join( //
          enhanceLayer, //
          optLinearChain.create(2 * numFreq, 1), SingletonLayer.VectorToScalar);
      return new NetModelData(new NetModel(netLayer, SquaredErrorLoss.INSTANCE), trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(2);
      {
        Show show = new Show("original");
        show.add(ImagePlot.of(ImageFormat.of(image)));
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        Show show = new Show("approximation");
        Tensor image2 = Meshgrid.of(networkDomain, dimension).image(netLayer::forward);
        show.add(ImagePlot.of(ImageFormat.of(image2.multiply(xFF).maps(Clips.positive(255)))));
        // show.add(ListPlot.of(trainingData.xdata()));
        show.render_autoIndent(graphics, list.get(1));
      }
    };
  }
}
