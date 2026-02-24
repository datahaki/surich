// code by jph
package ch.alpine.subare.book.ch02.bandits2;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

/* package */ enum BanditsHelper {
  ;
  static DiscreteQsa getOptimalQsa(StandardModel standardModel) {
    return ActionValueIteration.solve(standardModel, Chop._04);
  }
}
