// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.fish;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;

/** action value iteration for cliff walk */
enum AVI_Fishfarm {
  ;
  static void main() throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster fishfarmRaster = new FishfarmRaster(fishfarm);
    DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    // Export.of(UserHome.Pictures("cliffwalk_qsa_avi.png"), //
    // StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(fishfarm);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.of(fishfarm, ref, avi.qsa());
      imageIconRecorder.write(StateRasters.qsaLossRef(fishfarmRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isErrorFree())
        break;
    }
    // DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    // vs.print();
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, ref);
    // Policies.print(policy, cliffwalk.states());
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
