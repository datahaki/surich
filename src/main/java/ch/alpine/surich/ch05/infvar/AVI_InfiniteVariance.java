// code by jph
package ch.alpine.surich.ch05.infvar;

import ch.alpine.bridge.pro.VoidProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

class AVI_InfiniteVariance implements VoidProvider {
  @Override
  public Void runStandalone() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    DiscreteQsa qsa = ActionValueIteration.solve(infiniteVariance, Chop._05);
    DiscreteUtils.print(qsa);
    return null;
  }

  static void main() {
    new AVI_InfiniteVariance().runStandalone();
  }
}
