// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;

import ch.alpine.ascony.io.ImageIconRecorder;
import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.api.EpisodeVsEstimator;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.mc.ConstantAlphaMonteCarloVs;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.gfx.StateRasters;

@ReflectionMarker
class CAMC_Gridworld implements ManipulateProvider { // TODO SUBARE work in progress?
  @Override
  public Container getContainer() {
    Gridworld gridworld = new Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    // final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    EpisodeVsEstimator camc = ConstantAlphaMonteCarloVs.create( //
        gridworld, DefaultLearningRate.of(3, .51));
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(250);
    final int batches = 50;
    // Tensor epsilon = Subdivide.of(.2, .05, batches);
    for (int index = 0; index < batches; ++index) {
      System.out.println(index);
      for (int count = 0; count < 20; ++count) {
        Policy policy = EquiprobablePolicy.create(gridworld);
        // EGreedyPolicy.bestEquiprobable(gridworld, camc.vs(), epsilon.Get(index));
        ExploringStarts.batch(gridworld, policy, camc);
      }
      imageIconRecorder.write(StateRasters.vs(gridworldRaster, DiscreteValueFunctions.rescaled(camc.vs())));
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new CAMC_Gridworld().runStandalone();
  }
}
