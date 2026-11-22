// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

@ReflectionMarker
public class OptParam {
  public NetOptimizers GD = NetOptimizers.ADAM_W;
  public Scalar eta = RealScalar.of(0.003);
  public Scalar beta1 = RealScalar.of(0.9);
  public Scalar beta2 = RealScalar.of(0.99);

  public NetOptimizer create(NetLayer netLayer) {
    return GD.create(netLayer, eta, beta1, beta2);
  }
}
