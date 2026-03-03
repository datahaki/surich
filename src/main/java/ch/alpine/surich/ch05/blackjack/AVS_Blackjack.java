// code by jph
package ch.alpine.surich.ch05.blackjack;

import java.awt.Container;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.ActionValueStatistics;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Subdivide;
import ch.alpine.tensor.sca.Chop;

/** finding optimal policy to stay or hit
 * 
 * Figure 5.3 p.108 */
@ReflectionMarker
class AVS_Blackjack implements ManipulateProvider {
  public Integer batches = 5; // 40

  @Override
  public Container getContainer() {
    Blackjack blackjack = new Blackjack();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(blackjack);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    Tensor epsilon = Subdivide.of(.2, .05, batches);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(blackjack, mces.qsa(), sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.1, 0.01));
    int episodes = 0;
    ActionValueStatistics avs = new ActionValueStatistics(blackjack);
    for (int index = 0; index < batches; ++index) {
      System.out.println(index + " " + epsilon.Get(index));
      for (int count = 0; count < batches; ++count) {
        episodes += ExploringStarts.batchWithReplay(blackjack, policy, mces, avs);
      }
      ActionValueIteration avi = ActionValueIteration.of(blackjack, avs);
      avi.untilBelow(Chop._04, 3);
      imageIconRecorder.write( //
          Join.of( //
              BlackjackHelper.joinAll(blackjack, mces.qsa()), //
              BlackjackHelper.joinAll(blackjack, avi.qsa())));
      System.out.println(episodes + " " + avs.coverage());
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new AVS_Blackjack().runStandalone();
  }
}
