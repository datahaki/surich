// code by jph
package ch.alpine.surich.fish;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.surich.util.gfx.StateRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.sca.Round;

@ReflectionMarker
class RSTQP_Fishfarm implements ManipulateProvider {
  public Integer period = 10;
  public Integer max_fish = 10;
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    Fishfarm fishfarm = new Fishfarm(period, max_fish);
    FishfarmRaster cliffwalkRaster = new FishfarmRaster(fishfarm);
    final DiscreteQsa ref = FishfarmHelper.getOptimalQsa(fishfarm);
    DiscreteQsa qsa = DiscreteQsa.build(fishfarm, DoubleScalar.POSITIVE_INFINITY);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        fishfarm, qsa, ConstantLearningRate.one());
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(fishfarm, ref, qsa);
      TabularSteps.batch(fishfarm, rstqp);
      imageIconRecorder.write(StateRasters.qsaLossRef(cliffwalkRaster, qsa, ref));
      if (infoline.isLossfree())
        break;
    }
    DiscreteUtils.print(qsa, Round._2);
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new RSTQP_Fishfarm().runStandalone();
  }
}
