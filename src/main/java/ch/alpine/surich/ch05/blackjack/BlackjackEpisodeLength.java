// code by jph
package ch.alpine.surich.ch05.blackjack;

import java.awt.Container;
import java.util.Map;
import java.util.TreeMap;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.EpisodeKickoff;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.sca.exp.Log;

@ReflectionMarker
class BlackjackEpisodeLength implements ManipulateProvider {
  public Integer episodes = 100_000;
  private final Blackjack blackjack = new Blackjack();

  @Override
  public Container getContainer() {
    Policy policy = EquiprobablePolicy.create(blackjack);
    Map<Integer, Integer> counts = new TreeMap<>();
    for (int episode = 0; episode < episodes; ++episode) {
      EpisodeInterface episodeInterface = EpisodeKickoff.single(blackjack, policy);
      int count = 0;
      while (episodeInterface.hasNext()) {
        episodeInterface.step();
        ++count;
      }
      counts.merge(count, 1, Integer::sum);
    }
    Tensor xy = Tensor.of(counts.entrySet().stream() //
        .map(e -> Tensors.of(RealScalar.of(e.getKey()), Log.FUNCTION.apply(RealScalar.of(e.getValue())))));
    Show show = new Show();
    show.add(ListLinePlot.of(xy));
    return ShowGridComponent.of(show);
  }

  static void main() {
    new BlackjackEpisodeLength().runStandalone();
  }
}
