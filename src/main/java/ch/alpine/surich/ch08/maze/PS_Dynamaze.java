// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.PrioritizedSweeping;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.StepExploringStarts;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;

/** determines q(s, a) function for equiprobable "random" policy */
enum PS_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze2";
    Dynamaze dynamaze;
    // dynamaze = DynamazeHelper.original(name);
    dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(7, 1.01);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping( //
        sarsa, 10, RealScalar.ZERO);
    StepExploringStarts stepExploringStarts = //
        new StepExploringStarts(dynamaze, prioritizedSweeping) {
          @Override
          public Policy batchPolicy(int batch) {
            return policy;
          }
        };
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    while (stepExploringStarts.batchIndex() < batches) {
      Infoline infoline = Infoline.of(dynamaze, ref, qsa);
      stepExploringStarts.nextEpisode();
      imageIconRecorder.write(StateRasters.qsaLossRef(dynamazeRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    // handle(SarsaType.original, 10);
    // handle(SarsaType.expected, 50);
    handle(SarsaType.QLEARNING, 10);
  }
}
