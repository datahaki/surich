// code by jph
package ch.alpine.subare.book.ch04.grid;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIteration.solve(gridworld, Chop._04);
  }
}
