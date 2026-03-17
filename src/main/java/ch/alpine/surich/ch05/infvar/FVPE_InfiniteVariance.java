// code by jph
package ch.alpine.surich.ch05.infvar;

import ch.alpine.bridge.pro.VoidProvider;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.sca.N;

class FVPE_InfiniteVariance implements VoidProvider {
  @Override
  public Void runStandalone() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        infiniteVariance, null);
    Policy policy = new ConstantPolicy(Rational.of(5, 10));
    for (int count = 0; count < 100; ++count)
      ExploringStarts.batch(infiniteVariance, policy, fvpe);
    System.out.println(fvpe.vs().values().maps(N.DOUBLE));
    return null;
  }

  static void main() {
    new FVPE_InfiniteVariance().runStandalone();
  }
}
