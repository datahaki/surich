// code by jph
package ch.alpine.surich.ch05.blackjack;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class BlackjackTest {
  @SuppressWarnings("unused")
  @Test
  void testSimple() {
    Blackjack blackjack = new Blackjack();
    // TODO SUBARE fail sometimes, correct or wrong?
    {
      Tensor next = blackjack.move(Tensors.vector(0, 18, 7), RealScalar.ONE);
      // assertEquals(next, Tensors.vector(-1));
    }
    {
      Tensor next = blackjack.move(Tensors.vector(0, 21, 7), RealScalar.ZERO);
      // assertEquals(next, Tensors.vector(1));
    }
  }
}
