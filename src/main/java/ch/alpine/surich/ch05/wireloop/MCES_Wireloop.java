// code by jph
package ch.alpine.surich.ch05.wireloop;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.val.DiscreteQsa;

@ReflectionMarker
class MCES_Wireloop implements ManipulateProvider {
  public Integer batches = 1;

  @Override
  public Container getContainer() {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(wireloop);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(200));
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, mces.qsa(), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(wireloop, ref, mces.qsa());
      for (int count = 0; count < 4; ++count)
        ExploringStarts.batch(wireloop, policy, mces);
      imageIconRecorder.write(WireloopHelper.render(wireloopRaster, ref, mces.qsa()));
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new MCES_Wireloop().runStandalone();
  }
}
