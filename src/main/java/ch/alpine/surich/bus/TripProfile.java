// code by jph
package ch.alpine.surich.bus;

import ch.alpine.tensor.Scalar;

interface TripProfile {
  int length();

  Scalar costPerUnit(int time);

  Scalar unitsDrawn(int time);
}
