// code by jph
package ch.alpine.surich.ch06.cliff;

import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.pol.PolicyBase;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.surich.util.gfx.StateActionRasters;

/** monte carlo is bad in this example, since the steep negative reward biases most episodes */
// TODO SURICH this does not really converge at all
enum MCES_Cliffwalk {
  ;
  static void main() throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(cliffwalk);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    int batches = 100;
    for (int index = 0; index < batches; ++index) {
      Infoline.of(cliffwalk, ref, mces.qsa());
      for (int count = 0; count < 10; ++count) {
        StateActionCounter sac = new DiscreteStateActionCounter();
        PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(cliffwalk, mces.qsa(), sac);
        ExploringStarts.batch(cliffwalk, policy, mces);
      }
      imageIconRecorder.write(StateActionRasters.qsaLossRef(cliffwalkRaster, mces.qsa(), ref));
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
