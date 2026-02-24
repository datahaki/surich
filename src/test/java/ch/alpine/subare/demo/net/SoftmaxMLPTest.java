// code by jph
package ch.alpine.subare.demo.net;

import static org.junit.jupiter.api.Assertions.fail;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.demo.net.SoftmaxMLP.Network;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class SoftmaxMLPTest {
  @Test
  void test() {
    Tensor X = Tensors.matrixDouble(new double[][] { //
        { 1, 1 }, { 1.5, 2 }, { 2, 1 }, // Class 0
        { 5, 5 }, { 6, 5 }, { 5, 6 }, // Class 1
        { 8, 1 }, { 9, 2 }, { 8, 2 } // Class 2
    }).unmodifiable();
    Tensor y = Tensors.vectorInt(new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 }).unmodifiable();
    for (int attempt = 0; attempt < 3; ++attempt) {
      SoftmaxMLP softmaxMLP = new SoftmaxMLP();
      Network network = softmaxMLP.new Network();
      network.train(X, y);
      Scalar error = network.evaluate(X, y);
      if (Scalars.isZero(error))
        return;
    }
    fail();
  }
}
