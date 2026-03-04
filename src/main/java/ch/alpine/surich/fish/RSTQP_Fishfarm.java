// code by jph
package ch.alpine.surich.fish;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.sca.Round;

enum RSTQP_Fishfarm {
  ;
  static void main() throws Exception {
    Fishfarm fishfarm = new Fishfarm(20, 20);
    FishfarmRaster cliffwalkRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        fishfarm, qsa, ConstantLearningRate.one());
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    int batches = 20;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(fishfarm, ref, qsa);
      TabularSteps.batch(fishfarm, fishfarm, rstqp);
      imageIconRecorder.write(StateRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    DiscreteUtils.print(qsa, Round._2);
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
