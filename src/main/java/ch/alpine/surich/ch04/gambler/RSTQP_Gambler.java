// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;
import java.awt.GridLayout;
import java.time.Duration;

import javax.swing.JPanel;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.util.ActionValueStatistics;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.subare.val.DiscreteValueFunctions;
import ch.alpine.tensor.Rational;

// R1STQP algorithm is not suited for gambler's dilemma
@ReflectionMarker
class RSTQP_Gambler implements ManipulateProvider {
  public Integer batches = 200;

  @Override
  public Container getContainer() {
    GamblerModel gamblerModel = new GamblerModel(20, Rational.of(4, 10));
    GamblerRaster gamblerRaster = new GamblerRaster(gamblerModel);
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of(gamblerModel, qsa, //
        DefaultLearningRate.of(4, 0.71));
    ActionValueStatistics avs = new ActionValueStatistics(gamblerModel);
    ImageIconRecorder imageIconRecorder1 = ImageIconRecorder.loop(Duration.ofMillis(250));
    ImageIconRecorder imageIconRecorder2 = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int index = 0; index < batches; ++index) {
      Infoline infoline = Infoline.of(gamblerModel, ref, qsa);
      TabularSteps.batch(gamblerModel, rstqp, avs);
      imageIconRecorder1.write(StateActionRasters.qsaPolicyRef(gamblerRaster, qsa, ref));
      imageIconRecorder2.write(StateActionRasters.qsa( //
          gamblerRaster, DiscreteValueFunctions.rescaled(((DiscreteStateActionCounter) rstqp.sac()).inQsa(gamblerModel))));
      if (infoline.isLossfree())
        break;
    }
    // ---
    // ActionValueIteration avi = new ActionValueIteration(gambler, avs);
    // avi.setMachinePrecision();
    // avi.untilBelow(RealScalar.of(.0001));
    // Scalar error = DiscreteValueFunctions.distance(ref, avi.qsa());
    // System.out.println(error);
    // Export.of(UserHome.Pictures("gambler_avs.png"),
    // // GamblerHelper.qsaPolicyRef(gambler, avi.qsa(), ref)
    // StateActionRasters.qsaPolicyRef(new GamblerRaster(gambler), qsa, ref));
    JPanel jPanel = new JPanel(new GridLayout(2, 1));
    jPanel.add(AwtUtil.iconAsLabel(imageIconRecorder1.getIconImage()));
    jPanel.add(AwtUtil.iconAsLabel(imageIconRecorder2.getIconImage()));
    return jPanel;
  }

  static void main() {
    new RSTQP_Gambler().runStandalone();
  }
}
