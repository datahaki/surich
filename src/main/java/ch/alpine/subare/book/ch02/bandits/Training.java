// code by jph
package ch.alpine.subare.book.ch02.bandits;

import java.util.Map;
import java.util.Map.Entry;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.book.ch02.Agent;
import ch.alpine.subare.book.ch02.EGreedyAgent;
import ch.alpine.subare.book.ch02.GradientAgent;
import ch.alpine.subare.book.ch02.OptimistAgent;
import ch.alpine.subare.book.ch02.RandomAgent;
import ch.alpine.subare.book.ch02.UCBAgent;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;

/** chapter 2:
 * Multi-arm Bandits */
enum Training implements ShowProvider {
  INSTANCE;

  static Judger train(int epochs) {
    final int n = 3;
    Scalar econst = Rational.of(1, 12);
    Judger judger = new Judger(new Bandits(n), //
        new RandomAgent(n), //
        new GradientAgent(n, RealScalar.of(.1)), //
        new EGreedyAgent(n, _ -> econst, econst.toString()), //
        new EGreedyAgent(n, i -> Rational.of(1, Scalars.intValueExact(i) + 1), "1/i"), new UCBAgent(n, RealScalar.of(1)), //
        new UCBAgent(n, RealScalar.of(1.2)), //
        new UCBAgent(n, RealScalar.of(0.8)), //
        // new GradientAgent(n, 0.25), //
        new OptimistAgent(n, RealScalar.of(1), RealScalar.of(0.1)) //
    );
    // ---
    for (int round = 0; round < epochs; ++round)
      judger.play();
    return judger;
  }

  @Override
  public Show getShow() {
    Judger judger = train(100);
    judger.ranking();
    Map<Agent, Tensor> map = judger.map();
    Show show = new Show();
    for (Entry<Agent, Tensor> entry : map.entrySet()) {
      Showable showable = show.add(ListLinePlot.of(Range.of(0, entry.getValue().length()), entry.getValue()));
      showable.setLabel(entry.getKey().toString());
    }
    return show;
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
