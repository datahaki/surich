// code by jph
package ch.alpine.surich.ch04.grid;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.td.TabularTemporalDifference0;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.val.DiscreteValueFunctions;
import ch.alpine.subare.val.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

/** Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -9.00
 * {0, 2} -19.90
 * {0, 3} -20.52
 * {1, 0} -13.61
 * {1, 1} -16.52
 * {1, 2} -17.52
 * {1, 3} -19.84
 * {2, 0} -16.20
 * {2, 1} -17.77
 * {2, 2} -19.94
 * {2, 3} -11.45
 * {3, 0} -21.01
 * {3, 1} -19.68
 * {3, 2} -18.52
 * {3, 3} 0 */
// TODO does not seem to converge to anything
class TTD0_Gridworld implements ShowProvider {
  @Override
  public Show getShow() {
    Ch04Gridworld gridWorld = new Ch04Gridworld();
    DiscreteVs sol = ValueIteration.solve(gridWorld, Chop._04);
    DiscreteVs vs = DiscreteVs.build(gridWorld.states());
    TabularTemporalDifference0 ttd0 = new TabularTemporalDifference0( //
        vs, gridWorld.gamma(), DefaultLearningRate.of(3, .6), new DiscreteStateActionCounter());
    Policy policy = EquiprobablePolicy.create(gridWorld);
    TableBuilder tableBuilder = new TableBuilder();
    for (int count = 0; count < 300; ++count) {
      ExploringStarts.batch(gridWorld, policy, ttd0);
      Scalar diff = DiscreteValueFunctions.distance(sol, vs);
      tableBuilder.appendRow(RealScalar.of(count), diff);
    }
    DiscreteUtils.print(vs, Round._2);
    Show show = new Show();
    show.add(ListLinePlot.of(tableBuilder.getColumns(0, 1)));
    return show;
  }

  static void main() {
    new TTD0_Gridworld().runStandalone();
  }
}
