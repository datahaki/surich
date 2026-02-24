// code by jph
package ch.alpine.subare.book.ch04.gambler;

import java.nio.file.Path;

import javax.swing.JComponent;

import ch.alpine.bridge.fig.ArrayPlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.HomeDirectory;
import ch.alpine.tensor.img.ColorDataGradients;

/** action value iteration for gambler's dilemma
 * 
 * visualizes the exact optimal policy */
@ReflectionMarker
public class AVI_Gambler implements ManipulateProvider {
  Path path = HomeDirectory.Ephemeral.mk_dirs(AVI_Gambler.class.getSimpleName());
  public Integer max = 100;
  public Scalar P_win = Rational.THIRD;
  public ColorDataGradients cdg = ColorDataGradients.CLASSIC;

  @Override
  public JComponent getContainer() {
    GamblerModel gamblerModel = new GamblerModel(max, P_win);
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    Tensor qsa = StateActionRasters._render1(gamblerRaster, ref);
    Show show1 = new Show();
    show1.add(ArrayPlot.of(qsa, cdg));
    Tensor qsaPolicy = StateActionRasters._render2(gamblerRaster, ref);
    Show show2 = new Show();
    show2.add(ArrayPlot.of(qsaPolicy, cdg));
    DiscreteVs vs = DiscreteUtils.createVs(gamblerModel, ref);
    // try {
    // Put.of(path.resolve("ex403_vs_values"), vs.values());
    // } catch (IOException e) {
    // e.printStackTrace();
    // }
    return ShowGridComponent.of(show1, show2);
  }

  static void main() {
    new AVI_Gambler().runStandalone();
  }
}
