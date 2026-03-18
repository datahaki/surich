// code by jph
package ch.alpine.surich;

import java.util.List;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import ch.alpine.subare.alg.ActionValueIteration;
import ch.alpine.subare.alg.IterativePolicyEvaluation;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.mod.StandardModel;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.tensor.sca.Chop;

class SubareExperimentalTest {
  static List<StandardModel> standardModels() {
    return SubareExperimental.filter(StandardModel.class);
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
}
