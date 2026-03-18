// code by jz
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Tensor;

/** Q-Learning applied to gambler with adaptive learning rate */
@ReflectionMarker
class QL_Gambler implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.QLEARNING;
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    GamblerModel gamblerModel = new GamblerModel(20, 0.4);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    Policy policy = EquiprobablePolicy.create(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policyEGreedy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    policyEGreedy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    // System.out.println(qsa.size());
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.51);
    Sarsa stepDigest = sarsaType.sarsa(gamblerModel, learningRate, qsa, sac, policyEGreedy);
    for (int index = 0; index < batches; ++index) {
      Infoline.of(gamblerModel, ref, qsa);
      for (int count = 0; count < 1; ++count) {
        ExploringStarts.batch(gamblerModel, policy, 1, stepDigest);
      }
      imageIconRecorder.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), qsa, ref));
    }
    // DiscreteUtils.print(qsa, Round._2);
    // System.out.println("---");
    EpisodeInterface mce = EpisodeKickoff.single(gamblerModel, policy);
    while (mce.hasNext()) {
      StepRecord stepRecord = mce.step();
      Tensor state = stepRecord.prevState();
      System.out.println(state + " then " + stepRecord.action());
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new QL_Gambler().runStandalone();
  }
}
