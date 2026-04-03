// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.awt.ColumnPanel;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DequeExploringStarts;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LinearExplorationRate;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.surich.util.gfx.StateActionRasters;
import ch.alpine.tensor.io.TableBuilder;

/** 1, or N-step Original/Expected Sarsa, and QLearning for gridworld
 * 
 * covers Example 4.1, p.82 */
@ReflectionMarker
class SES_Gridworld implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Integer nstep = 1;
  public Integer batches = 3;

  @Override
  public Container getContainer() {
    Ch04Gridworld gridworld = new Ch04Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gridworld, qsa, sac);
    policy.setExplorationRate(LinearExplorationRate.of(batches, 0.2, 0.01));
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    LearningRate learningRate = DefaultLearningRate.of(5, 1.1);
    Sarsa sarsa = sarsaType.sarsa(gridworld, learningRate, qsa, sac, policy);
    DequeExploringStarts exploringStartsStream = new DequeExploringStarts(gridworld, nstep, sarsa) {
      @Override
      public Policy batchPolicy(int batch) {
        return policy;
      }
    };
    int episode = 0;
    TableBuilder tableBuilder = new TableBuilder();
    while (exploringStartsStream.batchIndex() < batches) {
      exploringStartsStream.nextEpisode();
      Infoline infoline = Infoline.of(gridworld, ref, qsa);
      tableBuilder.appendRow(infoline.indexedVector(episode));
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
      if (infoline.isLossfree())
        break;
      ++episode;
    }
    ColumnPanel columnPanel = new ColumnPanel();
    columnPanel.add(AwtUtil.iconAsLabel(imageIconRecorder.getIconImage()));
    {
      Show show = new Show();
      show.add(ListLinePlot.of(tableBuilder.getColumns(0, 1)));
      show.add(ListLinePlot.of(tableBuilder.getColumns(0, 2)));
      columnPanel.add(ShowGridComponent.of(show));
    }
    return columnPanel;
  }

  static void main() throws Exception {
    new SES_Gridworld().runStandalone();
  }
}
