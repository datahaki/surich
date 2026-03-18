// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.fish;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.subare.val.DiscreteQsa;

/** action value iteration for cliff walk */
@ReflectionMarker
class AVI_Fishfarm implements ManipulateProvider {
  public Integer period = 10;
  public Integer max_fish = 10;
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    Fishfarm fishfarm = new Fishfarm(period, max_fish);
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
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new AVI_Fishfarm().runStandalone();
  }
}
