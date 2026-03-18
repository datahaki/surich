// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch04.grid;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

/** determines value function for equiprobable "random" policy
 * 
 * Example 4.1, p.82
 * Figure 4.1, p.83
 * 
 * {0, 0} 0
 * {0, 1} -14.0
 * {0, 2} -20.0
 * {0, 3} -22.0
 * {1, 0} -14.0
 * {1, 1} -18.0
 * {1, 2} -20.0
 * {1, 3} -20.0
 * {2, 0} -20.0
 * {2, 1} -20.0
 * {2, 2} -18.0
 * {2, 3} -14.0
 * {3, 0} -22.0
 * {3, 1} -20.0
 * {3, 2} -14.0
 * {3, 3} 0 */
class IPE_Gridworld implements ShowProvider {
  @Override
  public Show getShow() {
    Gridworld gridworld = new Gridworld();
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation(gridworld, EquiprobablePolicy.create(gridworld));
    ipe.until(Chop._04);
    IO.println(ipe.iterations());
    DiscreteUtils.print(ipe.vs(), Round._1);
    Show show = new Show();
    show.add(ListLinePlot.of(ipe.tableBuilder().getColumns(0, 1)));
    return show;
  }

  static void main() {
    new IPE_Gridworld().runStandalone();
  }
}
