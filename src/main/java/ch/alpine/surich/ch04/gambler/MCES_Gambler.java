// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Round;

@ReflectionMarker
class MCES_Gambler implements ManipulateProvider {
  public Integer batches = 10;

  @Override
  public Container getContainer() {
    GamblerModel gambler = GamblerModel.createDefault();
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, mces.qsa(), sac);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    for (int index = 0; index < batches; ++index) {
      Infoline.of(gambler, ref, mces.qsa());
      ExploringStarts.batch(gambler, policy, mces);
      Tensor tensor = StateActionRasters.qsaPolicyRef(gamblerRaster, mces.qsa(), ref);
      imageIconRecorder.write(tensor);
    }
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    DiscreteUtils.print(discreteVs, Round._2);
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new MCES_Gambler().runStandalone();
  }
}
