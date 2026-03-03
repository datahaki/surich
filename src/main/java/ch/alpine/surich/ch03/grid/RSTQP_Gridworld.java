// code by jph
package ch.alpine.surich.ch03.grid;

import ch.alpine.bridge.fig.ListLinePlot;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.api.StepDigest;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.exp.Log;

class RSTQP_Gridworld implements ShowProvider {
  private static final TensorUnaryOperator LOGY = row -> Tensors.of(row.Get(0), Log.FUNCTION.apply(row.Get(1)));

  @Override
  public Show getShow() {
    Gridworld gridworld = new Gridworld();
    DiscreteQsa ref = ActionValueIteration.solve(gridworld, Chop._04);
    DiscreteQsa qsa = DiscreteQsa.build(gridworld);
    StepDigest stepDigest = Random1StepTabularQPlanning.of(gridworld, qsa, ConstantLearningRate.one());
    TableBuilder tableBuilder = new TableBuilder();
    for (int index = 0; index < 20; ++index) {
      Infoline infoline = Infoline.of(gridworld, ref, qsa);
      tableBuilder.appendRow(infoline.indexedVector(index));
      TabularSteps.batch(gridworld, gridworld, stepDigest);
      if (infoline.isLossfree())
        break;
    }
    Show show = new Show();
    tableBuilder.getTable();
    show.add(ListLinePlot.of(LOGY.slash(tableBuilder.getColumns(0, 1)))).setLabel("error");
    show.add(ListLinePlot.of(LOGY.slash(tableBuilder.getColumns(0, 2)))).setLabel("loss");
    return show;
  }

  static void main() {
    new RSTQP_Gridworld().runStandalone();
  }
}
