// code by jph
package ch.alpine.surich.ch05.infvar;

import ch.alpine.subare.pol.Policy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.d.CategoricalDistribution;

record ConstantPolicy(Scalar backProb) implements Policy {
  @Override
  public Scalar probability(Tensor state, Tensor action) {
    if (state.equals(RealScalar.ZERO))
      return action.equals(RealScalar.ZERO) //
          ? backProb
          : RealScalar.ONE.subtract(backProb);
    return RealScalar.ONE;
  }

  @Override
  public Distribution getDistribution(Tensor state) {
    Tensor actions = InfiniteVariance.INSTANCE.actions(state);
    Tensor pdf = Tensor.of(actions.stream() //
        .map(action -> probability(state, action)));
    return CategoricalDistribution.fromUnscaledPDF(pdf);
  }
}
