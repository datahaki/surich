// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch03.grid;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

/** produces results on p.64-65:
 * 
 * {0, 0} 3.3
 * {0, 1} 8.8
 * {0, 2} 4.4
 * {0, 3} 5.3
 * {0, 4} 1.5
 * {1, 0} 1.5
 * {1, 1} 3.0
 * {1, 2} 2.3
 * {1, 3} 1.9
 * {1, 4} 0.5
 * {2, 0} 0.1
 * {2, 1} 0.7
 * {2, 2} 0.7
 * {2, 3} 0.4
 * {2, 4} -0.4
 * {3, 0} -1.0
 * {3, 1} -0.4
 * {3, 2} -0.4
 * {3, 3} -0.6
 * {3, 4} -1.2
 * {4, 0} -1.9
 * {4, 1} -1.3
 * {4, 2} -1.2
 * {4, 3} -1.4
 * {4, 4} -2.0 */
class IPE_Gridworld implements ShowProvider {
  @Override
  public Show getShow() {
    Gridworld gridworld = new Gridworld();
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        gridworld, EquiprobablePolicy.create(gridworld));
    ipe.until(Chop._04);
    DiscreteUtils.print(ipe.vs(), Round._1);
    Show show = new Show();
    show.add(ListLinePlot.of(ipe.tableBuilder().getTable()));
    return show;
  }

  static void main() {
    new IPE_Gridworld().runStandalone();
  }
}
