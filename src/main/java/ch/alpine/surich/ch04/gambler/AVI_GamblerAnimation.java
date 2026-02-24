// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

import javax.swing.JLabel;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;

/** action value iteration for gambler's dilemma
 * 
 * visualizes each pass of the action value iteration */
@ReflectionMarker
enum AVI_GamblerAnimation implements ManipulateProvider {
  INSTANCE;

  private final JLabel jLabel;

  private AVI_GamblerAnimation() {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    ActionValueIteration avi = ActionValueIteration.of(gamblerModel);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(500);
    for (int index = 0; index < 13; ++index) {
      DiscreteQsa qsa = avi.qsa();
      Infoline.print(gamblerModel, index, ref, qsa);
      imageIconRecorder.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), qsa, ref));
      avi.step();
    }
    imageIconRecorder.write(StateActionRasters.qsaPolicyRef(new GamblerRaster(gamblerModel), avi.qsa(), ref));
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
