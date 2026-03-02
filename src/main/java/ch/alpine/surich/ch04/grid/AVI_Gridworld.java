// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** solving grid world
 * gives the value function for the optimal policy equivalent to
 * shortest path to terminal state
 * 
 * Example 4.1, p.82
 * 
 * {0, 0} 0
 * {0, 1} -1
 * {0, 2} -2
 * {0, 3} -3
 * {1, 0} -1
 * {1, 1} -2
 * {1, 2} -3
 * {1, 3} -2
 * {2, 0} -2
 * {2, 1} -3
 * {2, 2} -2
 * {2, 3} -1
 * {3, 0} -3
 * {3, 1} -2
 * {3, 2} -1
 * {3, 3} 0 */
enum AVI_Gridworld implements ManipulateProvider {
  INSTANCE;

  @Override
  public Container getContainer() {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int count = 0; count < 7; ++count) {
      imageIconRecorder.write(StateActionRasters.qsa(gridworldRaster, DiscreteValueFunctions.rescaled(avi.qsa())));
      avi.step();
    }
    // DiscreteUtils.print(avi.qsa());
    // DiscreteVs dvs = DiscreteUtils.createVs(gridworld, avi.qsa());
    // DiscreteUtils.print(dvs);
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
