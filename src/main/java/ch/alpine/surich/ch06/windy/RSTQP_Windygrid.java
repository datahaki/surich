// code by jph
package ch.alpine.surich.ch06.windy;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** the R1STQP algorithm is cheating on the Windygrid
 * because TabularSteps starts in every state-action pair
 * instead of only the 1 start state of Windygrid */
@ReflectionMarker
enum RSTQP_Windygrid implements ManipulateProvider {
  INSTANCE;

  @Override
  public Container getContainer() {
    Windygrid windygrid = Windygrid.createFour();
    WindygridRaster windygridRaster = new WindygridRaster(windygrid);
    final DiscreteQsa ref = WindygridHelper.getOptimalQsa(windygrid);
    DiscreteQsa qsa = DiscreteQsa.build(windygrid);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        windygrid, qsa, ConstantLearningRate.one());
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    int batches = 40;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(windygrid, ref, qsa);
      TabularSteps.batch(windygrid, windygrid, rstqp);
      imageIconRecorder.write(StateActionRasters.qsaLossRef(windygridRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
