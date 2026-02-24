// code by jph
package ch.alpine.surich.bus;

import java.io.IOException;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;
import ch.alpine.tensor.sca.Chop;

/* package */ enum AVI_Charger {
  ;
  static void main() throws IOException {
    TripProfile tripProfile = new ConstantDrawTrip(24, 3);
    Charger charger = new Charger(tripProfile, 6);
    DiscreteQsa ref = ActionValueIteration.solve(charger, Chop._04);
    ChargerRaster chargerRaster = new ChargerRaster(charger);
    Export.of(HomeDirectory.Pictures.resolve("charger_qsa_avi.png"), //
        StateActionRasters.qsaPolicy(chargerRaster, ref));
    // ref.print();
  }
}
