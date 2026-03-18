// code by jph
package ch.alpine.surich.ch06.maxbias;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

enum MaxbiasHelper {
  ;
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIteration.solve(maxbias, Chop._04);
  }
}
