// code by jph
package ch.alpine.surich.net;

import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.ElementwiseLayer;
import ch.alpine.subare.net.LinearLayer;
import ch.alpine.subare.net.NetChain;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.c.UniformDistribution;
import ch.alpine.tensor.sca.Clips;

class NetChainTest {
  @Test
  void test() {
    Distribution DISTRIBUTION = UniformDistribution.of(Clips.absolute(0.5));
    NetChain nc1 = NetChain.of(LinearFLayer.logSig(DISTRIBUTION, new Random(3), 3, 2));
    NetChain nc2 = NetChain.of( //
        LinearLayer.of(DISTRIBUTION, new Random(3), 3, 2), //
        ElementwiseLayer.logSig());
    Tolerance.CHOP.requireClose(nc1.parameters(), nc2.parameters());
    Tensor x = Tensors.vector(1, 2);
    Tensor y1 = nc1.forward(x);
    Tensor y2 = nc2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    Tensor d = Tensors.vector(0.2, 0.4, 0.6);
    Tensor g1 = nc1.back(d);
    Tensor g2 = nc2.back(d);
    IO.println(g1);
    IO.println(g2);
    nc1.update();
    nc2.update();
    // IO.println(nc1.parameters());
    // IO.println(nc2.parameters());
  }
}
