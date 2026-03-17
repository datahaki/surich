// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** Example 4.1, p.82 */
@ReflectionMarker
class MCES_Gridworld implements ManipulateProvider {
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, mces.qsa(), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.05));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      Infoline.of(gridworld, ref, mces.qsa());
      for (int count = 0; count < 1; ++count) {
        ExploringStarts.batch(gridworld, policy, mces);
      }
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), mces.qsa(), ref));
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new MCES_Gridworld().runStandalone();
  }
}
