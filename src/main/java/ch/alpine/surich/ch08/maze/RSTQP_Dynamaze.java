// code by jph
package ch.alpine.surich.ch08.maze;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.ConstantLearningRate;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

/**  */
@ReflectionMarker
class RSTQP_Dynamaze implements ManipulateProvider {
  public AVH_Dynamazes maze = AVH_Dynamazes.START_0;
  public Integer batches = 50;

  @Override
  public Container getContainer() {
    Dynamaze dynamaze = DynamazeHelper.create5(3);
    DynamazeRaster dynamazeRaster = new DynamazeRaster(dynamaze);
    DiscreteQsa ref = DynamazeHelper.getOptimalQsa(dynamaze);
    DiscreteQsa qsa = DiscreteQsa.build(dynamaze);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        dynamaze, qsa, ConstantLearningRate.of(RealScalar.ONE));
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(dynamaze, ref, qsa);
      TabularSteps.batch(dynamaze, rstqp);
      imageIconRecorder.write(StateRasters.vs_rescale(dynamazeRaster, qsa));
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new RSTQP_Dynamaze().runStandalone();
  }
}
