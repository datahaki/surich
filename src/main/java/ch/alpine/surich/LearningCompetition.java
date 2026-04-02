// code by jph
package ch.alpine.surich;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

import javax.swing.ImageIcon;

import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.subare.rate.ConstantExplorationRate;
import ch.alpine.subare.rate.ExplorationRate;
import ch.alpine.subare.util.Infoline;
import ch.alpine.subare.util.LearningContender;
import ch.alpine.subare.util.gfx.IntPoint;
import ch.alpine.subare.val.DiscreteQsa;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.api.ScalarTensorFunction;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.red.Min;

public class LearningCompetition {
  private final Map<IntPoint, LearningContender> map = new HashMap<>();
  private final ScalarTensorFunction colorDataFunction = ColorDataGradients.CLASSIC;
  // ---
  private final DiscreteQsa ref;
  private final Tensor epsilon;
  private final Scalar errorcap;
  private final Scalar errorcap2;
  // ---
  // override default values if necessary:
  public int period = 200;
  public int nstep = 1;

  public LearningCompetition(DiscreteQsa ref, Tensor epsilon, Scalar errorcap, Scalar errorcap2) {
    this.ref = ref;
    this.epsilon = epsilon.unmodifiable();
    this.errorcap = errorcap;
    this.errorcap2 = errorcap2;
  }

  public void put(IntPoint point, LearningContender learningContender) {
    map.put(point, learningContender);
  }

  private int RESX = 0;

  // TODO SURICH this should be shown in plots!?
  public ImageIcon doit() {
    RESX = map.keySet().stream().mapToInt(point -> point.x()).reduce(Math::max).orElseThrow() + 1;
    int RESY = map.keySet().stream().mapToInt(point -> point.y()).reduce(Math::max).orElseThrow() + 1;
    Tensor image = Array.zeros(RESX + 1 + RESX, RESY, 4);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(period));
    for (int index = 0; index < epsilon.length(); ++index) {
      final int findex = index;
      map.entrySet().stream().parallel().forEach(entry -> //
      processEntry(image, entry.getKey(), entry.getValue(), findex));
      // System.out.printf("%3d %s %n", index, timing.seconds().maps(Round._1));
      imageIconRecorder.write(image);
    }
    return imageIconRecorder.getIconImage();
  }

  private void processEntry(Tensor image, IntPoint point, LearningContender learningContender, int index) {
    ExplorationRate explorationRate = ConstantExplorationRate.of(epsilon.Get(index));
    learningContender.stepAndCompare(explorationRate, nstep, ref);
    Infoline infoline = learningContender.infoline(ref);
    {
      Scalar error = infoline.error();
      error = Min.of(error.divide(errorcap), RealScalar.ONE);
      image.set(colorDataFunction.apply(error), point.x(), point.y());
    }
    {
      Scalar error = infoline.loss();
      error = Min.of(error.divide(errorcap2), RealScalar.ONE);
      image.set(colorDataFunction.apply(error), RESX + 1 + point.x(), point.y());
    }
  }
}
