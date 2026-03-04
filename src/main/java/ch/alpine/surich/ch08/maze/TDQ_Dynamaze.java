// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TabularDynaQ;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateRasters;

/** determines q(s, a) function for equiprobable "random" policy */
enum TDQ_Dynamaze {
  ;
  static void handle(SarsaType sarsaType, int batches) throws Exception {
    System.out.println(sarsaType);
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(5, 0.51);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    TabularDynaQ tabularDynaQ = new TabularDynaQ(sarsa, 10);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      // if (EPISODES - 10 < index)
      Infoline.of(dynamaze, ref, qsa);
      // for (int count = 0; count < 5; ++count)
      ExploringStarts.batch(dynamaze, policy, tabularDynaQ);
      imageIconRecorder.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    // handle(SarsaType.original, 3, 50);
    // handle(SarsaType.expected, 2, 50);
    handle(SarsaType.QLEARNING, 50);
  }
}
