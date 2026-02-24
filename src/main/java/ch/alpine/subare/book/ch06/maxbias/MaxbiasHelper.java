// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

enum MaxbiasHelper {
  ;
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIteration.solve(maxbias, Chop._04);
  }
}
