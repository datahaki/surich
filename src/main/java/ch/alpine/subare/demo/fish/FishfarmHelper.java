// code by jph
package ch.alpine.subare.demo.fish;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

enum FishfarmHelper {
  ;
  static DiscreteQsa getOptimalQsa(Fishfarm cliffwalk) {
    return ActionValueIteration.solve(cliffwalk, Chop._04);
  }
  // static Policy getOptimalPolicy(Fishfarm cliffwalk) {
  // ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
  // vi.untilBelow(RealScalar.of(1e-10));
  // return GreedyPolicy.bestEquiprobable(cliffwalk, vi.vs());
  // }
}
