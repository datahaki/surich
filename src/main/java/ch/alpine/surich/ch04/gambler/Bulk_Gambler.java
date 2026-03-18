// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

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
import ch.alpine.subare.util.gfx.IntPoint;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.surich.LearningCompetition;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Subdivide;

/** Sarsa applied to gambler for different learning rate parameters */
@ReflectionMarker
class Bulk_Gambler implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.QLEARNING;
  public Integer nstep = 1;

  @Override
  public Container getContainer() {
    GamblerModel gamblerModel = new GamblerModel(20, Rational.of(4, 10)); // 20, 4/10
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(20); // 15
    final Scalar losscap = RealScalar.of(.25); // .5
    final Tensor epsilon = Subdivide.of(.2, .01, 100); // .2, .6
    int x = 0;
    String name = "gambler_Q_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep;
    LearningCompetition learningCompetition = new LearningCompetition(ref, epsilon, errorcap, losscap);
    learningCompetition.nstep = nstep;
    // TODO this should be shown in plots!
    for (Tensor factor : Subdivide.of(.1, 10, 8)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.3, 8)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(100, 0.2, 0.01));
        Sarsa sarsa = sarsaType.sarsa(gamblerModel, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(gamblerModel, sarsa);
        learningCompetition.put(new IntPoint(x, y), learningContender);
        ++y;
      }
      ++x;
    }
    // ---
    return AwtUtil.iconAsLabel(learningCompetition.doit());
  }

  static void main() throws Exception {
    new Bulk_Gambler().runStandalone();
  }
}
