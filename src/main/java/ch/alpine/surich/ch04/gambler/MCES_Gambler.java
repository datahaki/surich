// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

import javax.swing.ImageIcon;
import javax.swing.JLabel;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.sca.Round;

/* package */ enum MCES_Gambler implements ManipulateProvider {
  INSTANCE;

  private final JLabel jLabel;

  MCES_Gambler() {
    GamblerModel gambler = GamblerModel.createDefault();
    GamblerRaster gamblerRaster = new GamblerRaster(gambler);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gambler);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(gambler);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gambler, mces.qsa(), sac);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(200);
    int batches = 20;
    for (int index = 0; index < batches; ++index) {
      Infoline.print(gambler, index, ref, mces.qsa());
      ExploringStarts.batch(gambler, policy, mces);
      Tensor tensor = StateActionRasters.qsaPolicyRef(gamblerRaster, mces.qsa(), ref);
      imageIconRecorder.write(tensor);
    }
    ImageIcon iconImage = imageIconRecorder.getIconImage();
    System.out.println("done");
    DiscreteVs discreteVs = DiscreteUtils.createVs(gambler, mces.qsa());
    DiscreteUtils.print(discreteVs, Round._2);
    jLabel = AwtUtil.iconAsLabel(iconImage);
  }

  @Override
  public Container getContainer() {
    return jLabel;
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
