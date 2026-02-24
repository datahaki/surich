// code by jph
package ch.alpine.surich.ch04.grid;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.Chop;

class GridworldTest {
  @Test
  void testVI() {
    Gridworld gridworld = new Gridworld();
    ValueIteration vi = new ValueIteration(gridworld, gridworld);
    vi.untilBelow(Chop._04);
    DiscreteVs vs = vi.vs();
    // vs.print();
    assertEquals(vs.value(Tensors.vector(0, 2)), RealScalar.of(-2));
    assertEquals(vs.value(Tensors.vector(2, 1)), RealScalar.of(-3));
    assertEquals(vs.value(Tensors.vector(2, 3)), RealScalar.of(-1));
    assertEquals(vs.value(Tensors.vector(3, 3)), RealScalar.of(0));
  }

  @Test
  void testAVI() {
    Gridworld gridworld = new Gridworld();
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    avi.untilBelow(Chop._04);
    // ---
    DiscreteQsa qsa = avi.qsa();
    // qsa.print();
    assertEquals(qsa.value(Tensors.vector(0, 1), Tensors.vector(0, 1)), RealScalar.of(-3));
    assertEquals(qsa.value(Tensors.vector(1, 3), Tensors.vector(-1, 0)), RealScalar.of(-4));
    assertEquals(qsa.value(Tensors.vector(3, 3), Tensors.vector(1, 0)), RealScalar.of(0));
  }
}
