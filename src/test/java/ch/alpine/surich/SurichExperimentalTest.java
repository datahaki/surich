// code by jph
package ch.alpine.surich;

import java.util.List;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.alg.Random1StepTabularQPlanning;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.mc.FirstVisitPolicyEvaluation;
import ch.alpine.subare.mod.DiscreteModel;
import ch.alpine.subare.mod.MonteCarloInterface;
import ch.alpine.subare.mod.StandardModel;
import ch.alpine.subare.mod.TabularModel;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.subare.pol.StepDigest;
import ch.alpine.subare.rate.ConstantLearningRate;
import ch.alpine.subare.util.TabularSteps;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.sca.Chop;

class SurichExperimentalTest {
  static List<DiscreteModel> discreteModels() {
    return SurichExperimental.filter(DiscreteModel.class);
  }

  static List<StandardModel> standardModels() {
    return SurichExperimental.filter(StandardModel.class);
  }

  static List<TabularModel> tabularModels() {
    return SurichExperimental.filter(TabularModel.class);
  }

  // MonteCarloInterface
  static List<MonteCarloInterface> monteCarlos() {
    return SurichExperimental.filter(MonteCarloInterface.class);
  }

  @ParameterizedTest
  @MethodSource("standardModels")
  void testValueIteration(StandardModel standardModel) {
    ValueIteration.solve(standardModel, Chop._04);
    ActionValueIteration.solve(standardModel, Chop._04);
  }

  @ParameterizedTest
  @MethodSource("standardModels")
  void testIterativePolicyEvaluation(StandardModel standardModel) {
    IterativePolicyEvaluation ipe = new IterativePolicyEvaluation( //
        standardModel, //
        EquiprobablePolicy.create(standardModel));
    ipe.until(Chop._04);
  }

  @ParameterizedTest
  @MethodSource("tabularModels")
  void testRandom1(TabularModel tabularModel) {
    DiscreteQsa qsa = DiscreteQsa.build(tabularModel);
    StepDigest stepDigest = Random1StepTabularQPlanning.of(tabularModel, qsa, ConstantLearningRate.one());
    for (int index = 0; index < 20; ++index)
      TabularSteps.batch(tabularModel, stepDigest);
  }

  @ParameterizedTest
  @MethodSource("monteCarlos")
  void testFirstVisit(MonteCarloInterface discreteModel) {
    // TODO create random policy
    // Policy policy = PolicyType.GREEDY.bestEquiprobable(discreteModel, ref, null);
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        discreteModel, null);
    // for (int count = 0; count < 100; ++count)
    // ExploringStarts.batch(discreteModel, policy, fvpe);
  }
}
