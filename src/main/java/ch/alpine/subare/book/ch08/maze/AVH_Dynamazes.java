// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch08.maze;

import javax.swing.ImageIcon;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateRasters;

enum AVH_Dynamazes {
  START_0,
  START_1,
  START_2;

  ImageIcon iconImage;

  AVH_Dynamazes() {
    Dynamaze dynamaze = DynamazeHelper.create5(ordinal() + 1);
    DiscreteQsa est = DynamazeHeuristic.create(dynamaze);
    // est = DiscreteQsa.build(dynamaze);
    ActionValueIteration avi = ActionValueIteration.of(dynamaze, est);
    // ---
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(500);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    for (int index = 0; index < 50; ++index) {
      Infoline infoline = Infoline.print(dynamaze, index, ref, avi.qsa());
      imageIconRecorder.write(StateRasters.qsaLossRef(dynamazeRaster, avi.qsa(), ref));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    iconImage = imageIconRecorder.getIconImage();
  }
}
