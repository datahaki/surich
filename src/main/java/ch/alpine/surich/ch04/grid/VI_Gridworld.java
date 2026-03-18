// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch04.grid;

import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.pol.Policies;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.subare.val.DiscreteValueFunctions;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.io.Export;
import ch.alpine.tensor.sca.Chop;

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
enum VI_Gridworld {
  ;
  static void main() throws Exception {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldStateRaster = new GridworldRaster(gridworld);
    ValueIteration vi = new ValueIteration(gridworld);
    vi.untilBelow(Chop._04);
    DiscreteUtils.print(vi.vs());
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gridworld, vi.vs(), null);
    Policies.print(policy, gridworld.states());
    Export.of(HomeDirectory.Pictures.resolve("gridworld_vs_vi.png"), //
        StateRasters.vs(gridworldStateRaster, DiscreteValueFunctions.rescaled(vi.vs())));
    // GridworldHelper.render(gridworld, vi.vs())
  }
}
