// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch06.cliff;

import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
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
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.sca.Round;

/** StepDigest qsa methods applied to cliff walk */
enum Sarsa_Cliffwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk, DoubleScalar.POSITIVE_INFINITY);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(cliffwalk, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    Sarsa sarsa = sarsaType.sarsa(cliffwalk, DefaultLearningRate.of(7, 0.61), qsa, sac, policy);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      // if (batches - 10 < index)
      Infoline infoline = Infoline.of(cliffwalk, ref, qsa);
      ExploringStarts.batch(cliffwalk, policy, nstep, sarsa);
      imageIconRecorder.write(StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
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
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    // handle(SarsaType.original, 1, 30);
    // handle(SarsaType.expected, 1, 30);
    handle(SarsaType.QLEARNING, 1, 10);
  }
}
