// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.TrainingData;
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
class Interp1Param {
  public static final Clip DOMAIN = Clips.absoluteOne();
  public Integer dom = 5;
  public Integer samples = 30;

  TrainingData generate() {
    Tensor disY = RandomVariate.of(NormalDistribution.standard(), dom);
    ScalarTensorFunction stf = BSplineFunctionString.of(3, disY);
    ScalarUnaryOperator lin = LinearInterpolation.of(Clips.positive(dom - 1));
    Tensor xdata = RandomVariate.of(UniformDistribution.of(DOMAIN), samples, 1);
    ScalarUnaryOperator suo = s -> (Scalar) stf.apply(lin.apply(DOMAIN.rescale(s)));
    Tensor ydata = xdata.maps(suo);
    return new TrainingData(xdata, ydata);
  }
}
