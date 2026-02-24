// code by jph
package ch.alpine.surich.net;

import java.util.random.RandomGenerator;

import ch.alpine.subare.net.Layer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.Ramp;
import ch.alpine.tensor.sca.UnitStep;
import ch.alpine.tensor.sca.exp.DLogisticSigmoid;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

public class LinearFLayer implements Layer {
  public static LinearFLayer logSig(Distribution d, RandomGenerator randomGenerator, int ante, int post) {
    LinearFLayer linearLayer = new LinearFLayer();
    linearLayer.f = LogisticSigmoid.FUNCTION;
    linearLayer.df = DLogisticSigmoid.NESTED;
    linearLayer.W = RandomVariate.of(d, randomGenerator, ante, post);
    linearLayer.b = Array.zeros(ante);
    return linearLayer;
  }

  public static LinearFLayer reLu(Distribution d, int ante, int post) {
    LinearFLayer linearLayer = new LinearFLayer();
    linearLayer.f = Ramp.FUNCTION;
    linearLayer.df = UnitStep.FUNCTION;
    linearLayer.W = RandomVariate.of(d, ante, post);
    linearLayer.b = Array.zeros(ante);
    return linearLayer;
  }

  public static LinearFLayer maxE(Distribution d, int ante, int post) {
    LinearFLayer linearLayer = new LinearFLayer();
    linearLayer.f = s -> s;
    linearLayer.df = UnitStep.FUNCTION;
    linearLayer.W = RandomVariate.of(d, ante, post);
    linearLayer.b = Array.zeros(ante);
    return linearLayer;
  }

  ScalarUnaryOperator f;
  ScalarUnaryOperator df;
  Tensor W;
  Tensor b;
  Tensor inputCache;
  Tensor gW;
  Tensor gb;

  @Override
  public Tensor forward(Tensor x) {
    return W.dot(inputCache = x).add(b).maps(f);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = TensorProduct.of(gradOutput, inputCache);
    gb = gradOutput;
    return Entrywise.mul().apply(gradOutput.dot(W), inputCache.maps(df));
  }

  @Override
  public void update() {
    W = W.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    throw new RuntimeException();
  }

  @Override
  public Tensor parameters() {
    return Flatten.of(W, b);
  }
}
