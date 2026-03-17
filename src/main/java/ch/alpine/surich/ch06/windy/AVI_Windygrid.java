// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch06.windy;

import java.awt.Container;
import java.io.IOException;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.Policies;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;

/** action value iteration for cliff walk */
@ReflectionMarker
class AVI_Windygrid implements ManipulateProvider {
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    ActionValueIteration avi = ActionValueIteration.of(windygrid);
    try {
      Export.of(HomeDirectory.Pictures.resolve("windygrid_qsa_avi.png"), //
          StateActionRasters.qsa_rescaled(windygridRaster, ref));
    } catch (IOException e) {
      e.printStackTrace();
    }
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(windygrid, ref, avi.qsa());
      imageIconRecorder.write(StateActionRasters.qsaLossRef(windygridRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    // TODO SUBARE extract code below to other file
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, ref, null);
    Policies.print(policy, windygrid.states());
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new AVI_Windygrid().runStandalone();
  }
}
