// code by jph
package ch.alpine.surich.ch05.wireloop;

import java.awt.Container;

import javax.swing.JLabel;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.tensor.RealScalar;

/** Example 4.1, p.82 */
@ReflectionMarker
enum RSTQP_Wireloop implements ManipulateProvider {
  INSTANCE;

  private final JLabel jLabel;

  RSTQP_Wireloop() {
    String name = "wire5";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    DiscreteQsa qsa = DiscreteQsa.build(wireloop);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        wireloop, qsa, ConstantLearningRate.of(RealScalar.ONE));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    int batches = 50;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.print(wireloop, index, ref, qsa);
      TabularSteps.batch(wireloop, wireloop, rstqp);
      imageIconRecorder.write(WireloopHelper.render(wireloopRaster, ref, qsa));
      if (infoline.isLossfree())
        break;
    }
    jLabel = AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  @Override
  public Container getContainer() {
    return jLabel;
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
