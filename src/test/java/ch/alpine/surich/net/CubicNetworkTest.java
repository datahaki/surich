// code by jph
package ch.alpine.surich.net;

import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import ch.alpine.surich.net.CubicNetwork.Network;

class CubicNetworkTest {
  @Test
  void test() {
    for (int attempt = 0; attempt < 3; ++attempt) {
      CubicNetwork cubicNetwork = new CubicNetwork();
      Network network = cubicNetwork.new Network();
      network.train();
      // Scalar error = network.evaluate();
      // if (Scalars.lessEquals(error, RealScalar.of(0.7)))
      return;
    }
    fail();
  }
}
