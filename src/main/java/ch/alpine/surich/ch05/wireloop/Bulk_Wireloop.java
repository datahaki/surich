// code by jph
package ch.alpine.surich.ch05.wireloop;

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
class Bulk_Wireloop implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.QLEARNING;
  public Integer nstep = 1;

  @Override
  public Container getContainer() {
    String name = "wire4";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x); // 20, 4/10
    final DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop); // true q-function, for error measurement
    // ---
    final Scalar errorcap = RealScalar.of(15); // 15
    final Scalar losscap = RealScalar.of(.05); // .5
    final Tensor epsilon = Subdivide.of(.2, .05, 40); // .2, .6
    int x = 0;
    String name1 = name + "_Q_" + sarsaType.name() + "_E" + epsilon.Get(0) + "_N" + nstep;
    LearningCompetition learningCompetition = new LearningCompetition( //
        ref, epsilon, errorcap, losscap);
    learningCompetition.nstep = 1;
    // TODO SURICH resolution is low: 5 x 5
    for (Tensor factor : Subdivide.of(.1, 10, 5)) { // .5 16
      int y = 0;
      for (Tensor exponent : Subdivide.of(.51, 1.5, 5)) { // .51 2
        DiscreteQsa qsa = DiscreteQsa.build(wireloop);
        StateActionCounter sac = new DiscreteStateActionCounter();
        EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(wireloop, qsa, sac);
        policy.setExplorationRate(LinearExplorationRate.of(40, 0.2, 0.05));
        Sarsa sarsa = sarsaType.sarsa(wireloop, DefaultLearningRate.of((Scalar) factor, (Scalar) exponent), qsa, sac, policy);
        LearningContender learningContender = LearningContender.sarsa(wireloop, sarsa);
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
    new Bulk_Wireloop().runStandalone();
  }
}
