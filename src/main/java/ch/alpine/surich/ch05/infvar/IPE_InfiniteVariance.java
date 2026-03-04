// code by jph
package ch.alpine.surich.ch05.infvar;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

// TODO SUBARE check again
class IPE_InfiniteVariance implements ShowProvider {
  @Override
  public Show getShow() {
    StandardModel standardModel = new InfiniteVariance();
    Policy policy = new ConstantPolicy(Rational.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policy);
    a.until(Chop._04);
    DiscreteUtils.print(a.vs(), Round._2);
    Show show = new Show();
    show.add(ListLinePlot.of(a.tableBuilder().getTable()));
    return show;
  }

  static void main() {
    new IPE_InfiniteVariance().runStandalone();
  }
}
