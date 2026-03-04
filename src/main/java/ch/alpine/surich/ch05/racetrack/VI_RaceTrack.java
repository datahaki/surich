// code by jph
package ch.alpine.surich.ch05.racetrack;

import java.awt.Container;
import java.util.LinkedList;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.mc.MonteCarloEpisode;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.Primitives;
import ch.alpine.tensor.sca.Chop;

/** "track2", 4 */
@ReflectionMarker
class VI_RaceTrack implements ManipulateProvider {
  private final Racetrack racetrack;

  public VI_RaceTrack(String name, int maxSpeed) {
    racetrack = RacetrackHelper.create(name, maxSpeed);
  }

  @Override
  public Container getContainer() {
    ValueIteration vi = new ValueIteration(racetrack, racetrack);
    vi.untilBelow(Chop.below(10), 5);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(racetrack, vi.vs(), null);
    ImageIconRecorder imageIconRecorder = new ImageIconRecorder(400);
    for (Tensor start : racetrack.statesStart) {
      Tensor image = racetrack.image();
      MonteCarloEpisode mce = new MonteCarloEpisode(racetrack, policy, start, new LinkedList<>());
      while (mce.hasNext()) {
        StepRecord stepRecord = mce.step();
        {
          Tensor state = stepRecord.prevState();
          int[] index = Primitives.toIntArray(state);
          image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
        {
          Tensor state = stepRecord.nextState();
          int[] index = Primitives.toIntArray(state);
          image.set(Tensors.vector(128, 128, 128, 255), index[0], index[1]);
        }
      }
      imageIconRecorder.write(image);
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() throws Exception {
    new VI_RaceTrack("track2", 5).runStandalone();
  }
}
