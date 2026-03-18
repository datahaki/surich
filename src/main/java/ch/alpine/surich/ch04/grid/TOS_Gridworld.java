// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.awt.ColumnPanel;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowGridComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.pol.PolicyBase;
import ch.alpine.subare.api.pol.PolicyType;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TrueOnlineSarsa;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.qty.Timing;

@ReflectionMarker
class TOS_Gridworld implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Scalar LAMBDA = RealScalar.of(0.5);

  @Override
  public Container getContainer() {
    Gridworld gridworld = new Gridworld();
    final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    FeatureMapper mapper = ExactFeatureMapper.of(gridworld);
    FeatureWeight w = new FeatureWeight(mapper);
    // Tensor epsilon = Subdivide.of(.2, .01, batches); // used in egreedy
    // DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(gridworld, DiscreteQsa.build(gridworld), sac);
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gridworld, LAMBDA, mapper, learningRate, w, sac, policy);
    Timing timing = Timing.started();
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    TableBuilder tableBuilder = new TableBuilder();
    for (int batch = 0; batch < 100; ++batch) {
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      policy.setQsa(trueOnlineSarsa.qsaInterface());
      ExploringStarts.batch(gridworld, policy, trueOnlineSarsa);
      // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
      // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      DiscreteQsa qsa = trueOnlineSarsa.qsa();
      Infoline infoline = Infoline.of(gridworld, ref, qsa);
      tableBuilder.appendRow(infoline.indexedVector(batch));
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GridworldRaster(gridworld), qsa, ref));
      if (infoline.isLossfree()) {
        IO.println("lossfree after " + batch);
        break;
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + timing.seconds() + "s");
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
    new TOS_Gridworld().runStandalone();
  }
}
