// code by jph
package ch.alpine.subare.book.ch05.infvar;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.sca.Chop;

enum AVI_InfiniteVariance {
  ;
  static void main() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    DiscreteQsa qsa = ActionValueIteration.solve(infiniteVariance, Chop._05);
    DiscreteUtils.print(qsa);
  }
}
