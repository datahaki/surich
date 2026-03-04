// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;
import java.awt.GridLayout;

import javax.swing.JPanel;

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
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;

/** Sarsa applied to gambler */
@ReflectionMarker
class Sarsa_Gambler implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.QLEARNING;
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    GamblerModel gamblerModel = new GamblerModel(20, Rational.of(4, 10));
    /** true q-function, for error measurement */
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel); // q-function for training, initialized to 0
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    // ---
    ImageIconRecorder imageIconRecorder1 = new ImageIconRecorder(250);
    ImageIconRecorder imageIconRecorder2 = new ImageIconRecorder(250);
    Sarsa sarsa = sarsaType.sarsa(gamblerModel, learningRate, qsa, sac, policy);
    for (int index = 0; index < batches; ++index) {
      Infoline.of(gamblerModel, ref, qsa);
      ExploringStarts.batch(gamblerModel, policy, 1, sarsa);
      // ---
      imageIconRecorder1.write(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
      imageIconRecorder2.write(StateActionRasters.qsa( //
          gamblerRaster, DiscreteValueFunctions.rescaled(((DiscreteStateActionCounter) sarsa.sac()).inQsa(gamblerModel))));
    }
    // GamblerHelper.play(gamblerModel, qsa);
    JPanel jPanel = new JPanel(new GridLayout(2, 1));
    jPanel.add(AwtUtil.iconAsLabel(imageIconRecorder1.getIconImage()));
    jPanel.add(AwtUtil.iconAsLabel(imageIconRecorder2.getIconImage()));
    return jPanel;
  }

  static void main() {
    new Sarsa_Gambler().runStandalone();
  }
}
