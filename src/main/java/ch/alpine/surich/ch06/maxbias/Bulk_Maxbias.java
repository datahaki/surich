// code by jph
package ch.alpine.surich.ch06.maxbias;

import java.awt.Container;

import javax.swing.ImageIcon;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.LearningContender;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.surich.LearningCompetition;
import ch.alpine.surich.util.gfx.IntPoint;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
@ReflectionMarker
enum Bulk_Maxbias implements ManipulateProvider {
  INSTANCE(SarsaType.QLEARNING, 1);

  private final SarsaType sarsaType;
  private final int nstep;

  Bulk_Maxbias(SarsaType sarsaType, int nstep) {
    this.sarsaType = sarsaType;
    this.nstep = nstep;
  }

  @Override
  public Container getContainer() {
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
        learningCompetition.put(new IntPoint(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    ImageIcon imageIcon = learningCompetition.doit();
    return AwtUtil.iconAsLabel(imageIcon);
  }

  static void main() throws Exception {
    INSTANCE.runStandalone();
  }
}
