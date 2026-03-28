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
import ch.alpine.subare.td.DoubleSarsa;
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

/** Double Sarsa for gridworld */
@ReflectionMarker
class Double_Gridworld implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.ORIGINAL;
  public Integer nstep = 1;

  @Override
  public Container getContainer() {
    System.out.println("double " + sarsaType);
    Ch04Gridworld gridworld = new Ch04Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    int batches = 40;
    DiscreteQsa qsa1 = DiscreteQsa.build(gridworld);
    DiscreteQsa qsa2 = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    EGreedyPolicy policy1 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa1, sac1);
    policy1.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    EGreedyPolicy policy2 = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa2, sac2);
    policy2.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    DoubleSarsa doubleSarsa = sarsaType.doubleSarsa( //
        gridworld, //
        DefaultLearningRate.of(5, .51), //
        qsa1, qsa2, sac1, sac2, policy1, policy2);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      if (batches - 10 < index)
        Infoline.of(gridworld, ref, qsa1);
      policy.setQsa(doubleSarsa.qsa());
      policy.setSac(sac);
      ExploringStarts.batch(gridworld, policy, nstep, doubleSarsa);
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa1, ref));
    }
    // qsa.print(Round.toMultipleOf(DecimalScalar.of(.01)));
    System.out.println("---");
    DiscreteVs vs = DiscreteUtils.createVs(gridworld, doubleSarsa.qsa());
    Policy policyVs = PolicyType.GREEDY.bestEquiprobable(gridworld, vs, null);
    EpisodeInterface ei = EpisodeKickoff.single(gridworld, policyVs);
    while (ei.hasNext()) {
      StepRecord stepRecord = ei.step();
      Tensor state = stepRecord.prevState();
      System.out.println(state + " then " + stepRecord.action());
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new Double_Gridworld().runStandalone();
  }
}
