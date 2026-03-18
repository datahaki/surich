// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch03.grid;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.val.DiscreteVs;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Round;

/** solving grid world using action value iteration
 * 
 * produces results on p.71:
 * 
 * {0, 0} 22.0
 * {0, 1} 24.4
 * {0, 2} 22.0
 * {0, 3} 19.4
 * {0, 4} 17.5
 * {1, 0} 19.8
 * {1, 1} 22.0
 * {1, 2} 19.8
 * {1, 3} 17.8
 * {1, 4} 16.0
 * {2, 0} 17.8
 * {2, 1} 19.8
 * {2, 2} 17.8
 * {2, 3} 16.0
 * {2, 4} 14.4
 * {3, 0} 16.0
 * {3, 1} 17.8
 * {3, 2} 16.0
 * {3, 3} 14.4
 * {3, 4} 13.0
 * {4, 0} 14.4
 * {4, 1} 16.0
 * {4, 2} 14.4
 * {4, 3} 13.0
 * {4, 4} 11.7 */
class AVI_Gridworld implements ShowProvider {
  @Override
  public Show getShow() {
    Ch03Gridworld gridworld = new Ch03Gridworld();
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    avi.untilBelow(Chop._04);
    DiscreteUtils.print(avi.qsa(), Round._1);
    DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    DiscreteUtils.print(dvs, Round._1);
    Show show = new Show();
    show.add(ListLinePlot.of(avi.tableBuilder().getTable()));
    return show;
  }

  static void main() {
    new AVI_Gridworld().runStandalone();
  }
}
