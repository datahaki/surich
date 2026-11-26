// code by jph
package ch.alpine.surich.net.img;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ArrayPlot;
import ch.alpine.bridge.math.Meshgrid;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.SingletonLayer;
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
  public Integer samples = 20;
  public final OptLinearChain optLinearChain = new OptLinearChain();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    final Tensor image;
    final Meshgrid meshgrid;

    public InnerComponent() {
      super(seed);
      image = Import.of("ch/alpine/surich/net/img/gray_ma.jpg").divide(RealScalar.of(255));
      Dimension dimension = new Dimension(Unprotect.dimension1(image), image.length());
      Scalar redux = RealScalar.of((double) Math.max(dimension.width, dimension.height));
      meshgrid = Meshgrid.of(CoordinateBoundingBox.of( //
          Clips.positive(RealScalar.of(dimension.width).divide(redux)), //
          Clips.positive(RealScalar.of(dimension.height).divide(redux))), AwtUtil.aspect(samples, dimension));
      Tensor px = meshgrid.px_incr();
      Tensor py = meshgrid.py_decr();
      Tensor xdata = Tensors.reserve(samples * samples);
      Tensor ydata = Tensors.reserve(samples * samples);
      Interpolation interpolation = LinearInterpolation.of(image);
      for (Tensor x : px) {
        for (Tensor y : py) {
          Tensor p = Unprotect.byRef(y, x);
          xdata.append(p);
          ydata.append(interpolation.get(p.multiply(redux)));
        }
      }
      trainingData = new TrainingData(xdata, ydata);
    }

    @Override
    public NetModelData createNetModelDataBatch() {
      NetLayer netLayer = NetChain.join(optLinearChain.create(2, 1), SingletonLayer.VectorToScalar);
      return new NetModelData(new NetModel(netLayer, SquaredErrorLoss.INSTANCE), trainingData);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(2);
      {
        Show show = new Show("function");
        show.add(ArrayPlot.of(image));
        show.render_autoIndent(graphics, list.get(0));
      }
      {
        Show show = new Show("function");
        Tensor image2 = meshgrid.image(netLayer::forward);
        show.add(ArrayPlot.of(image2));
        show.render_autoIndent(graphics, list.get(1));
        // Show show = new Show("error");
        // ScalarBinaryOperator sbo = (x, y) -> {
        // Tensor p = Unprotect.using(List.of(x, y));
        // return (Scalar) netLayer.forward(p).subtract(truth.apply(p));
        // };
        // show.add(DensityPlot.of(sbo, ext, cdg));
        // show.add(PolygonPlot.of(cbb));
        // show.add(ListPlot.of(trainingData.xdata()));
        // show.render_autoIndent(graphics, list.get(1));
      }
    };
  }
}
