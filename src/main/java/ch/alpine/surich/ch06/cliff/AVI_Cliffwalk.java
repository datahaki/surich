// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch06.cliff;

import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.pol.Policies;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.subare.val.DiscreteValueFunctions;
import ch.alpine.subare.val.DiscreteVs;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;

/** action value iteration for cliff walk */
enum AVI_Cliffwalk {
  ;
  static void main() throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    Export.of(HomeDirectory.Pictures.resolve("cliffwalk_qsa_avi.png"), //
        StateActionRasters.qsa(new CliffwalkRaster(cliffwalk), DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(cliffwalk);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.of(cliffwalk, ref, avi.qsa());
      imageIconRecorder.write(StateActionRasters.qsaLossRef(cliffwalkRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    DiscreteVs vs = DiscreteUtils.createVs(cliffwalk, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(cliffwalk, ref, null);
    Policies.print(policy, cliffwalk.states());
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
