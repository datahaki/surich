// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.ConstantLearningRate;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;

/** Example 4.1, p.82 */
@ReflectionMarker
class RSTQP_Gridworld implements ManipulateProvider {
  public Integer batches = 10;

  @Override
  public Container getContainer() {
    Ch04Gridworld gridworld = new Ch04Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    Random1StepTabularQPlanning rstqp = //
        Random1StepTabularQPlanning.of(gridworld, qsa, ConstantLearningRate.one());
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
      Infoline.of(gridworld, ref, qsa);
      TabularSteps.batch(gridworld, rstqp);
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new RSTQP_Gridworld().runStandalone();
  }
}
