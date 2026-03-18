// code by jph
package ch.alpine.surich.ch06.cliff;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.sca.Round;

enum RSTQP_Cliffwalk {
  ;
  static void main() throws Exception {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    CliffwalkRaster cliffwalkRaster = new CliffwalkRaster(cliffwalk);
    final DiscreteQsa ref = CliffwalkHelper.getOptimalQsa(cliffwalk);
    DiscreteQsa qsa = DiscreteQsa.build(cliffwalk, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        cliffwalk, qsa, ConstantLearningRate.one());
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    int batches = 20;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(cliffwalk, ref, qsa);
      TabularSteps.batch(cliffwalk, rstqp);
      imageIconRecorder.write(StateActionRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    DiscreteUtils.print(qsa, Round._2);
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
