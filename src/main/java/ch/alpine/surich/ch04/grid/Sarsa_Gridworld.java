// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.epi.EpisodeInterface;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.subare.val.DiscreteVs;
import ch.alpine.tensor.Tensor;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
@ReflectionMarker
class Sarsa_Gridworld implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Integer nstep = 2;

  @Override
  public Container getContainer() {
    Ch04Gridworld gridworld = new Ch04Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 10;
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    Sarsa sarsa = sarsaType.sarsa(gridworld, learningRate, qsa, sac, policy);
    for (int index = 0; index < batches; ++index) {
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
      Infoline.of(gridworld, ref, qsa);
      // sarsa.supplyPolicy(() -> policy);
      ExploringStarts.batch(gridworld, policy, nstep, sarsa);
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, qsa);
    Policy policyVs = PolicyType.GREEDY.bestEquiprobable(gridworld, vs, null);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyVs);
    while (ei.hasNext()) {
      StepRecord stepRecord = ei.step();
      Tensor state = stepRecord.prevState();
      System.out.println(state + " then " + stepRecord.action());
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new Sarsa_Gridworld().runStandalone();
  }
}
