// code by jph
package ch.alpine.surich.ch05.wireloop;

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
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

/** Example 4.1, p.82 */
@ReflectionMarker
class RSTQP_Wireloop implements ManipulateProvider {
  public Integer batches = 20;

  @Override
  public Container getContainer() {
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        wireloop, qsa, ConstantLearningRate.of(RealScalar.ONE));
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(wireloop, ref, qsa);
      TabularSteps.batch(wireloop, rstqp);
      imageIconRecorder.write(WireloopHelper.render(wireloopRaster, ref, qsa));
      if (infoline.isLossfree())
        break;
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new RSTQP_Wireloop().runStandalone();
  }
}
