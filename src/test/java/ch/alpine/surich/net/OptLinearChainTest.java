// code by jph
package ch.alpine.surich.net;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.los.NetLoss;
import ch.alpine.subare.net.los.SquaredErrorLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.num.Pi;

class OptLinearChainTest {
  @Test
  void test() {
    OptLinearChain optLinearChain = new OptLinearChain();
    NetLayer netLayer = optLinearChain.asSUO();
    NetLayers.initialize(netLayer);
    NetModel netModel = new NetModel(netLayer, SquaredErrorLoss.INSTANCE);
    {
      Tensor y = netLayer.forward(Pi.VALUE);
      assertInstanceOf(Scalar.class, y);
      NetLoss netLoss = netModel.lossFunction().compute(y, RealScalar.TWO);
      Tensor gradOutput = netLoss.gradOutput();
      netLayer.back(gradOutput);
    }
    {
      Tensor y = netLayer.forwardBatch(Tensors.vector(1, 2, 3, 4));
      VectorQ.require(y);
      NetLoss netLoss = netModel.lossFunction().computeBatch(y, Tensors.vector(2, 3, 2, 1));
      Tensor gradOutput = netLoss.gradOutput();
      netLayer.backBatch(gradOutput);
    }
  }
}
