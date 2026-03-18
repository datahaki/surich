// code by jph
package ch.alpine.surich.ch05.racetrack;

import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.val.DiscreteVs;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.sca.Round;

enum FVPE_RaceTrack {
  ;
  static void main() {
    Racetrack racetrack = new Racetrack(Import.of("ch/alpine/subare/ch05/track0.png"), 3);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        racetrack, null);
    Policy policy = EquiprobablePolicy.create(racetrack);
    for (int count = 0; count < 10; ++count)
      ExploringStarts.batch(racetrack, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    DiscreteUtils.print(vs, Round._1);
  }
}
