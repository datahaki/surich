// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.PrioritizedSweeping;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.StepExploringStarts;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;

/** determines q(s, a) function for equiprobable "random" policy */
@ReflectionMarker
class PS_Dynamaze implements ManipulateProvider {
  public AVH_Dynamazes maze = AVH_Dynamazes.START_0;
  public SarsaType sarsaType = SarsaType.QLEARNING;
  public Integer batches = 10;

  @Override
  public Container getContainer() {
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    final DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    LearningRate learningRate = DefaultLearningRate.of(7, 1.01);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(dynamaze, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    Sarsa sarsa = sarsaType.sarsa(dynamaze, learningRate, qsa, sac, policy);
    PrioritizedSweeping prioritizedSweeping = new PrioritizedSweeping(sarsa, 10, RealScalar.ZERO);
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
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new PS_Dynamaze().runStandalone();
  }
}
