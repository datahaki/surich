// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Container;

import javax.swing.JPanel;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.awt.ColumnPanel;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.ShowComponent;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.pol.StateActionCounter;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.rate.LearningRate;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.td.TrueOnlineSarsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.gfx.StateActionRasters;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.subare.val.FeatureMapper;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.qty.Timing;

@ReflectionMarker
class TOS_Gambler implements ManipulateProvider {
  public SarsaType sarsaType = SarsaType.EXPECTED;
  public Scalar LAMBDA = RealScalar.of(0.3);
  public Integer batches = 100;

  @Override
  public Container getContainer() {
    GamblerModel gamblerModel = new GamblerModel(20, RealScalar.of(.4));
    final DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    FeatureMapper mapper = ExactFeatureMapper.of(gamblerModel);
    FeatureWeight w = new FeatureWeight(mapper);
    DiscreteQsa qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac);
    LearningRate learningRate = DefaultLearningRate.of(RealScalar.of(3), RealScalar.of(0.81));
    // LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.3), false); // the case without warmStart
    TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(gamblerModel, LAMBDA, mapper, learningRate, w, sac, policy);
    Timing timing = Timing.started();
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    TableBuilder tableBuilder = new TableBuilder();
    for (int batch = 0; batch < batches; ++batch) {
      // System.out.println("batch " + batch);
      policy.setQsa(trueOnlineSarsa.qsaInterface());
      ExploringStarts.batch(gamblerModel, policy, trueOnlineSarsa);
      // DiscreteQsa toQsa = trueOnlineSarsa.getQsa();
      // XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), errorAnalysis.getError(monteCarloInterface, optimalQsa, toQsa).number()));
      DiscreteQsa qsaRef = trueOnlineSarsa.qsa();
      Infoline infoline = Infoline.of(gamblerModel, ref, qsaRef);
      tableBuilder.appendRow(infoline.indexedVector(batch));
      imageIconRecorder.write(StateActionRasters.qsaLossRef(new GamblerRaster(gamblerModel), qsaRef, ref));
      if (infoline.isLossfree()) {
        break;
      }
    }
    System.out.println("Time for TrueOnlineSarsa: " + timing.seconds() + "s");
    JPanel jPanel = new ColumnPanel();
    jPanel.add(AwtUtil.iconAsLabel(imageIconRecorder.getIconImage()));
    Show show = new Show();
    show.add(ListLinePlot.of(tableBuilder.getColumns(0, 1)));
    show.add(ListLinePlot.of(tableBuilder.getColumns(0, 2)));
    ShowComponent showComponent = new ShowComponent();
    showComponent.setShow(show);
    jPanel.add(showComponent);
    return jPanel;
  }

  static void main() {
    new TOS_Gambler().runStandalone();
  }
}
