// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.fish;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
@ReflectionMarker
class Sarsa_Fishfarm implements ManipulateProvider {
  public Integer period = 10;
  public Integer max_fish = 10;
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Integer nstep = 1;
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    Fishfarm fishfarm = new Fishfarm(period, max_fish);
    FishfarmRaster fishfarmRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(fishfarm, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.5, 0.01));
    Sarsa sarsa = sarsaType.sarsa(fishfarm, DefaultLearningRate.of(7, 0.61), qsa, sac, policy);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      // if (batches - 10 < index)
      Infoline infoline = Infoline.of(fishfarm, ref, qsa);
      // sarsa.supplyPolicy(() -> policy);
      ExploringStarts.batch(fishfarm, policy, nstep, sarsa);
      imageIconRecorder.write(StateRasters.qsaLossRef(fishfarmRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    DiscreteUtils.print(qsa, Round._2);
    // System.out.println("---");
    // Policy policy = GreedyPolicy.bestEquiprobable(cliffwalk, qsa);
    // EpisodeInterface mce = EpisodeKickoff.single(cliffwalk, policy);
    // while (mce.hasNext()) {
    // StepInterface stepRecord = mce.step();
    // Tensor state = stepRecord.prevState();
    // System.out.println(state + " then " + stepRecord.action());
    // }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new Sarsa_Fishfarm().runStandalone();
  }
}
