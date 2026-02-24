// code by jph
package ch.alpine.subare.book.ch05.racetrack;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.sca.Round;

enum FVPE_RaceTrack {
  ;
  static void main() throws IOException {
    Path path = Unprotect.resourcePath("/ch/alpine/subare/ch05/track0.png");
    Throw.unless(Files.exists(path));
    Racetrack racetrack = new Racetrack(Import.of(path), 3);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        racetrack, null);
    Policy policy = EquiprobablePolicy.create(racetrack);
    for (int count = 0; count < 10; ++count)
      ExploringStarts.batch(racetrack, policy, fvpe);
    DiscreteVs vs = fvpe.vs();
    DiscreteUtils.print(vs, Round._1);
  }
}
