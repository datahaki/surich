// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.windy;

import java.awt.Container;
import java.io.IOException;

import javax.swing.JLabel;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.Policy;
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
enum AVI_Windygrid implements ManipulateProvider {
  INSANCE;

  private final JLabel jLabel;

  private AVI_Windygrid() {
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
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.print(windygrid, index, ref, avi.qsa());
      imageIconRecorder.write(StateActionRasters.qsaLossRef(windygridRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    // TODO SUBARE extract code below to other file
    jLabel = AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
    DiscreteVs vs = DiscreteUtils.createVs(windygrid, ref);
    DiscreteUtils.print(vs);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(windygrid, ref, null);
    Policies.print(policy, windygrid.states());
  }

  @Override
  public Container getContainer() {
    return jLabel;
  }

  static void main() {
    INSANCE.runStandalone();
  }
}
