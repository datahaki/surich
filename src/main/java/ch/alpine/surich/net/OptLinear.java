// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

@ReflectionMarker
public class OptLinear {
  public Tensor lin_dims = Tensors.vector(10, 6, 6);
  public ElementwiseLayers between = ElementwiseLayers.RELU;
  public Scalar dropout_p = RealScalar.ZERO;

  public NetLayer create(int d_in, int d_out) {
    int depth = lin_dims.length();
    int[] dims = new int[depth + 2];
    dims[0] = d_in;
    for (int i = 1; i <= depth; ++i)
      dims[i] = Scalars.intValueExact(lin_dims.Get(i - 1));
    dims[dims.length - 1] = d_out;
    return NetChains.linearWith(between, dropout_p, dims);
  }
}
