// code by jph
package ch.alpine.surich.ch08.maze;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;

/**  */
enum RSTQP_Dynamaze {
  ;
  static void main() throws Exception {
    String name = "maze5";
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        dynamaze, qsa, ConstantLearningRate.of(RealScalar.ONE));
    int batches = 50;
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(dynamaze, ref, qsa);
      TabularSteps.batch(dynamaze, dynamaze, rstqp);
      imageIconRecorder.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
      if (infoline.isLossfree())
        break;
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
