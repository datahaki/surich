// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DequeExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
@ReflectionMarker
class SES_Gridworld implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Integer nstep = 1;
  public Integer batches = 3;

  @Override
  public Container getContainer() {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
    Sarsa sarsa = sarsaType.sarsa(gridworld, learningRate, qsa, sac, policy);
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        return policy;
      }
    };
    int episode = 0;
    while (exploringStartsStream.batchIndex() < batches) {
      exploringStartsStream.nextEpisode();
      if (episode % 5 == 0) {
        Infoline infoline = Infoline.of(gridworld, ref, qsa);
        imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
        if (infoline.isLossfree())
          break;
      }
      ++episode;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new SES_Gridworld().runStandalone();
  }
}
