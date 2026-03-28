// code by jph
package ch.alpine.surich.ch04.rental;

import java.awt.Container;
import java.time.Duration;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.io.ani.ImageIconRecorder;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.alg.ValueIteration;

/** Example 4.2: Jack's Car Rental
 * Figure 4.2
 * 
 * p.87-88 */
@ReflectionMarker
class VI_CarRental implements ManipulateProvider {
  public Integer maxCars = 5;
  public Integer batches = 25;

  @Override
  public Container getContainer() {
    CarRental carRental = new CarRental(maxCars);
    ValueIteration vi = new ValueIteration(carRental);
    ImageIconRecorder imageIconRecorder = ImageIconRecorder.loop(Duration.ofMillis(250));
    for (int count = 0; count <= batches; ++count) {
      System.out.println(count);
      imageIconRecorder.write(CarRentalHelper.joinAll(carRental, vi.vs()));
      vi.step();
    }
    return AwtUtil.iconAsLabel(imageIconRecorder.getIconImage());
  }

  static void main() {
    new VI_CarRental().runStandalone();
  }
}
