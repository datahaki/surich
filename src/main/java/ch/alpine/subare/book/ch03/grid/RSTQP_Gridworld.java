// code by jph
package ch.alpine.subare.book.ch03.grid;

import ch.alpine.bridge.pro.RunProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.tensor.sca.Chop;

/* package */ enum RSTQP_Gridworld implements RunProvider {
  INSTANCE;

  @Override
  public Object runStandalone() {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIteration.solve(gridworld, Chop._04);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StepDigest stepDigest = Random1StepTabularQPlanning.of( //
        gridworld, qsa, ConstantLearningRate.one());
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.print(gridworld, index, ref, qsa);
      TabularSteps.batch(gridworld, gridworld, stepDigest);
      if (infoline.isLossfree())
        break;
    }
    return qsa;
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
