// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import java.awt.Container;
import java.awt.Point;

import javax.swing.ImageIcon;
import javax.swing.JLabel;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.book.LearningCompetition;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.LearningContender;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
@ReflectionMarker
enum Bulk_Maxbias implements ManipulateProvider {
  INSTANCE(SarsaType.QLEARNING, 1);

  private final JLabel jLabel;

  Bulk_Maxbias(SarsaType sarsaType, int nstep) {
    Maxbias maxbias = new Maxbias(1); // 20, 4/10
    final DiscreteQsa ref = MaxbiasHelper.getOptimalQsa(maxbias); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(.5); // 15
    final Scalar losscap = RealScalar.of(.5); // .5
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    String name = "maxbias_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    learningCompetition.period = 100;
    for (Tensor factor : Subdivide.of(.1, 10, 20)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 2, 10)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(maxbias);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(maxbias, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.2, 0.01));
        Sarsa sarsa = sarsaType.sarsa(maxbias, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(maxbias, sarsa);
        learningCompetition.put(new Point(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    ImageIcon imageIcon = learningCompetition.doit();
    jLabel = AwtUtil.iconAsLabel(imageIcon);
  }

  @Override
  public Container getContainer() {
    return jLabel;
  }

  static void main() throws Exception {
    INSTANCE.runStandalone();
  }
}
