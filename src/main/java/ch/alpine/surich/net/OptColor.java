// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.col.HuePalette;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.col.ColorDataIndexed;
import ch.alpine.tensor.col.ColorDataLists;

@ReflectionMarker
public class OptColor {
  public ColorDataLists cdl = ColorDataLists._097;
  public Scalar sat = RealScalar.of(0.5);
  public Scalar val = RealScalar.of(0.5);

  public ForeAndBackPalettes create() {
    ColorDataIndexed cdi = cdl.cyclic();
    ColorDataIndexed cdb = HuePalette.of(cdi).create(sat, val);
    return new ForeAndBackPalettes(cdi, cdb);
  }
}
