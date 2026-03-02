// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch06.windy;

import java.awt.Container;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** determines q(s, a) function for equiprobable "random" policy */
@ReflectionMarker
record Sarsa_Windygrid(SarsaType sarsaType, int batches) implements ManipulateProvider {
  public static final ManipulateProvider S1 = new Sarsa_Windygrid(SarsaType.ORIGINAL, 20);
  public static final ManipulateProvider S2 = new Sarsa_Windygrid(SarsaType.EXPECTED, 20);
  public static final ManipulateProvider S3 = new Sarsa_Windygrid(SarsaType.QLEARNING, 20);

  @Override
  public Container getContainer() {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    LearningRate learningRate = DefaultLearningRate.of(3, 0.51);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(windygrid, qsa, sac);
    Sarsa sarsa = sarsaType.sarsa(windygrid, learningRate, qsa, sac, policy);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(windygrid, ref, qsa);
      // sarsa.supplyPolicy(() -> policy);
      for (int count = 0; count < 10; ++count) // because there is only 1 start state
        ExploringStarts.batch(windygrid, policy, sarsa);
      imageIconRecorder.write(StateActionRasters.qsaLossRef(windygridRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    S1.runStandalone();
  }
}
