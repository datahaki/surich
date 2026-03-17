// code by jph
package ch.alpine.surich.ch05.wireloop;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.gfx.StateRasters;

@ReflectionMarker
class IPE_Wireloop implements ManipulateProvider {
  @Override
  public Container getContainer() {
    String name = "wire5";
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x);
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Policy policy = EquiprobablePolicy.create(wireloop);
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        wireloop, policy);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(200);
    for (int count = 0; count < 20; ++count) {
      imageIconRecorder.write(StateRasters.vs_rescale(wireloopRaster, ipe.vs()));
      for (int ep = 0; ep < 5; ++ep)
        ipe.step();
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new IPE_Wireloop().runStandalone();
  }
}
