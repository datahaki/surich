// code by jph
package ch.alpine.surich.ch04.grid;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.epi.EpisodeVsEstimator;
import ch.alpine.subare.mc.ConstantAlphaMonteCarloVs;
import ch.alpine.subare.pol.EquiprobablePolicy;
import ch.alpine.subare.pol.Policy;
import ch.alpine.subare.rate.DefaultLearningRate;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.subare.val.DiscreteValueFunctions;

@ReflectionMarker
class CAMC_Gridworld implements ManipulateProvider {
  // TODO SURICH work in progress?
  @Override
  public Container getContainer() {
    Ch04Gridworld gridworld = new Ch04Gridworld();
    GridworldRaster gridworldRaster = new GridworldRaster(gridworld);
    // final DiscreteQsa ref = GridworldHelper.getOptimalQsa(gridworld);
    EpisodeVsEstimator camc = ConstantAlphaMonteCarloVs.create( //
        gridworld, DefaultLearningRate.of(3, .51));
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
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
