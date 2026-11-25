// code by jph
package ch.alpine.surich.net;

import java.util.ArrayList;
import java.util.List;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

@ReflectionMarker
public class OptLinearChain {
  public Tensor lin_dims = Tensors.vector(8, 8);
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public Scalar dropout_p = RealScalar.ZERO;

  public NetLayer create(int d_in, int d_out) {
    List<Integer> list = new ArrayList<>();
    list.add(d_in);
    int depth = lin_dims.length();
    for (int i = 0; i < depth; ++i)
      list.add(Scalars.intValueExact(lin_dims.Get(i)));
    list.add(d_out);
    return NetChains.linearWith(list, between, dropout_p);
  }

  public NetLayer swiGLU(int d_in, int d_out) {
    int depth = lin_dims.Get(0).number().intValue();
    return new SwiGLULayer(d_in, depth, d_out);
  }

  public NetLayer asSUO() {
    return NetChain.join(ScalarLayer.ScalarToVector, create(1, 1), ScalarLayer.VectorToScalar);
  }
}
