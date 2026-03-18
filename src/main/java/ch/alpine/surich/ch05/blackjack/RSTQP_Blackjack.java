// code by jph
package ch.alpine.surich.ch05.blackjack;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.TabularSteps;

/** finding optimal policy to stay or hit
 * 
 * Random1StepTabularQPlanning does not seem to work on blackjack */
enum RSTQP_Blackjack {
  ;
  static void main() throws Exception {
    Blackjack blackjack = new Blackjack();
    DiscreteQsa qsa = DiscreteQsa.build(blackjack);
    Random1StepTabularQPlanning rstqp = Random1StepTabularQPlanning.of( //
        blackjack, qsa, DefaultLearningRate.of(5, 0.51));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    int batches = 60;
    for (int index = 0; index < batches; ++index) {
      for (int count = 0; count < 100; ++count)
        TabularSteps.batch(blackjack, rstqp);
      imageIconRecorder.write(BlackjackHelper.joinAll(blackjack, qsa));
    }
    AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }
}
