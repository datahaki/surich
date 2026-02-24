// code by jph
package ch.alpine.surich.ch03.grid;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.math.Index;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.surich.ch04.grid.Gridworld;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Chop;

class GridworldTest {
  @Disabled
  @Test
  void testBasics() {
    Gridworld gridworld = new Gridworld();
    assertEquals(gridworld.reward(Tensors.vector(0, 0), Tensors.vector(1, 0), null), RealScalar.ZERO);
    assertEquals(gridworld.reward(Tensors.vector(0, 0), Tensors.vector(-1, 0), null), RealScalar.ONE.negate());
  }

  @Disabled
  @Test
  void testIndex() {
    Gridworld gridworld = new Gridworld();
    Index actionsIndex = Index.build(gridworld.actions(null));
    int index = actionsIndex.of(Tensors.vector(1, 0));
    assertEquals(index, 3);
  }

  @Test
  void testR1STQL() {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIteration.solve(gridworld, Chop._04);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StepDigest stepDigest = //
        Random1StepTabularQPlanning.of(gridworld, qsa, ConstantLearningRate.of(RealScalar.ONE));
    Scalar error = null;
    for (int index = 0; index < 40; ++index) {
      TabularSteps.batch(gridworld, gridworld, stepDigest);
      error = DiscreteValueFunctions.distance(ref, qsa);
    }
    assertTrue(Scalars.lessThan(error, RealScalar.of(3)));
  }
}
