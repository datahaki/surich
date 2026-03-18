// code by jph
package ch.alpine.surich.ch06.cliff;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DequeExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
enum SES_Cliffwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep, int batches) throws Exception {
    System.out.println(sarsaType);
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    // CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    // Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk);
    LearningRate learningRate = DefaultLearningRate.of(7, 0.61);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(cliffwalk, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    Sarsa sarsa = sarsaType.sarsa(cliffwalk, learningRate, qsa, sac, policy);
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(cliffwalk, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        System.out.println("batch " + batch);
        return policy;
      }
    };
    int episode = 0;
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    while (exploringStartsStream.batchIndex() < batches) {
      exploringStartsStream.nextEpisode();
      // if (episode % 5 == 0)
      {
        Infoline infoline = Infoline.of(cliffwalk, ref, qsa);
        imageIconRecorder.write(StateActionRasters.qsaLossRef(new CliffwalkRaster(cliffwalk), qsa, ref));
        if (infoline.isLossfree())
          break;
      }
      ++episode;
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    int nstep = 1;
    // handle(SarsaType.original, nstep, 3);
    // handle(SarsaType.expected, nstep, 3);
    handle(SarsaType.QLEARNING, nstep, 10);
  }
}
