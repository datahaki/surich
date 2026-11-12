// code by jph
package ch.alpine.surich.net;

import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import ch.alpine.surich.net.XORNeuralNetwork.Network;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.exp.Exp;

class XORNeuralNetworkTest {
  @Test
  void test() {
    Tensor y = XORNeuralNetwork.XOR;
    for (int attempt = 0; attempt < 3; ++attempt) {
      XORNeuralNetwork xorNeuralNetwork = new XORNeuralNetwork();
      Network network = xorNeuralNetwork.new Network(y);
      Scalar error = Exp.FUNCTION.apply(network.netTrain.log_loss());
      // IO.println(error);
      if (Scalars.lessThan(error, RealScalar.of(0.02)))
        return;
    }
    fail();
  }
}
