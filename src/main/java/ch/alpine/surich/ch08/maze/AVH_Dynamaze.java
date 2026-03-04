// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;

@ReflectionMarker
class AVH_Dynamaze implements ManipulateProvider {
  public AVH_Dynamazes maze = AVH_Dynamazes.START_0;
  public Integer batches = 50;

  @Override
  public Container getContainer() {
    Dynamaze dynamaze = maze.dynamaze;
    DiscreteQsa est = DynamazeHeuristic.create(dynamaze);
    // est = DiscreteQsa.build(dynamaze);
    ActionValueIteration avi = ActionValueIteration.of(dynamaze, est);
    // ---
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(500);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(dynamaze, ref, avi.qsa());
      imageIconRecorder.write(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new AVH_Dynamaze().runStandalone();
  }
}
