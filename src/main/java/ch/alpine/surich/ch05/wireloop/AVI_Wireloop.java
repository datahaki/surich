// code by jph
package ch.alpine.surich.ch05.wireloop;

import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.val.DiscreteQsa;

enum AVI_Wireloop {
  ;
  static void main() throws Exception {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteQsa ref = WireloopHelper.getOptimalQsa(wireloop);
    ActionValueIteration avi = ActionValueIteration.of(wireloop);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    int batches = 50;
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(wireloop, ref, avi.qsa());
      imageIconRecorder.write(WireloopHelper.render(wireloopRaster, ref, avi.qsa()));
      avi.step();
      if (infoline.isLossfree())
        break;
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
