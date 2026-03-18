// code by jph
package ch.alpine.surich.ch06.cliff;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

enum CliffwalkHelper {
  ;
  static DiscreteQsa getOptimalQsa(Cliffwalk cliffwalk) {
    return ActionValueIteration.solve(cliffwalk, Chop._04);
  }

  static Policy getOptimalPolicy(Cliffwalk cliffwalk) {
    ValueIteration vi = new ValueIteration(cliffwalk);
    vi.untilBelow(Chop._10);
    return PolicyType.GREEDY.bestEquiprobable(cliffwalk, vi.vs(), null);
  }
}
