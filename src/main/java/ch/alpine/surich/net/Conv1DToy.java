// code by jph
package ch.alpine.surich.net;

import ch.alpine.subare.net.Conv1DLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

public class Conv1DToy {
  private final Conv1DLayer conv1dLayer = new Conv1DLayer();

  public Conv1DToy(int kernelSize) {
    conv1dLayer.w = RandomVariate.of(NormalDistribution.standard(), kernelSize);
    conv1dLayer.b = RandomVariate.of(NormalDistribution.standard());
  }

  public Tensor target(Tensor data) {
    TensorUnaryOperator tuo = conv1dLayer::forward;
    return tuo.slash(data);
  }

  public void check(Tensor X, Tensor Y) {
    TensorUnaryOperator tuo = conv1dLayer::forward;
    Tolerance.CHOP.requireClose(tuo.slash(X), Y);
  }
}
