// code by jph
package ch.alpine.surich.net.demo;

import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.sca.bes.BesselJ;
import ch.alpine.tensor.sca.ply.Chebyshev;

enum FeatureFunc {
  Chebyshev_T {
    @Override
    ScalarUnaryOperator create(int i) {
      return Chebyshev.T.of(i);
    }
  },
  Bessel_J {
    @Override
    ScalarUnaryOperator create(int i) {
      return x -> BesselJ.of(i, x);
    }
  };

  abstract ScalarUnaryOperator create(int i);
}
