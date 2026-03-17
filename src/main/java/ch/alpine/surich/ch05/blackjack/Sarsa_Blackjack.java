// code by jph
package ch.alpine.surich.ch05.blackjack;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.sca.Round;

// TODO SUBARE this demo throws an exception
/* package */ enum Sarsa_Blackjack {
  ;
  static void handle(SarsaType sarsaType) throws Exception {
    System.out.println(sarsaType);
    final Blackjack blackjack = new Blackjack();
    int batches = 40;
    Tensor epsilon = Subdivide.of(.1, .01, batches); // only used in egreedy
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    Sarsa sarsa = sarsaType.sarsa(blackjack, DefaultLearningRate.of(2, 0.6), qsa, sac, policy);
    for (int index = 0; index < batches; ++index) {
      // Scalar error = DiscreteQsas.distance(qsa, ref);
      System.out.println(index + " " + epsilon.Get(index).maps(Round._2));
      // sarsa.supplyPolicy(() -> policy);
      for (int count = 0; count < 10; ++count)
        ExploringStarts.batch(blackjack, policy, sarsa);
      imageIconRecorder.write(BlackjackHelper.joinAll(blackjack, qsa));
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    handle(SarsaType.QLEARNING);
  }
}
