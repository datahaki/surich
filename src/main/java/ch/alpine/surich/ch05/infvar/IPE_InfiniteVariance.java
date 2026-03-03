// code by jph
package ch.alpine.surich.ch05.infvar;

import ch.alpine.bridge.pro.VoidProvider;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Round;

// TODO SUBARE check again
class IPE_InfiniteVariance implements VoidProvider {
  @Override
  public Void runStandalone() {
    StandardModel standardModel = new InfiniteVariance();
    Policy policy = new ConstantPolicy(Rational.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policy);
    a.until(RealScalar.of(.0001));
    System.out.println(a.iterations());
    DiscreteUtils.print(a.vs(), Round._2);
    return null;
  }

  static void main() {
    new IPE_InfiniteVariance().runStandalone();
  }
}
