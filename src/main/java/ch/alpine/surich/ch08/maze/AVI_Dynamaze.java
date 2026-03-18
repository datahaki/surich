// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.pol.Policies;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;

/** action value iteration for cliff walk */
enum AVI_Dynamaze {
  ;
  static void main() throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    // Export.of(UserHome.Pictures("dynamaze_qsa_avi.png"), //
    // DynamazeHelper.render(windygrid, DiscreteValueFunctions.rescaled(ref)));
    ActionValueIteration avi = ActionValueIteration.of(dynamaze);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < 50; ++index) {
      Infoline infoline = Infoline.of(dynamaze, ref, avi.qsa());
      imageIconRecorder.write(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    // gsw.append(ImageFormat.of(DynamazeHelper.render(dynamaze, avi.qsa(), ref)));
    // TODO SUBARE extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(dynamaze, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(dynamaze, ref, null);
    Policies.print(policy, dynamaze.states());
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
