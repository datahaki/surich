// code by jph
package ch.alpine.surich.ch04.gambler;

import java.awt.Dimension;

import ch.alpine.subare.mod.DiscreteModel;
import ch.alpine.surich.util.gfx.IntPoint;
import ch.alpine.surich.util.gfx.StateActionRaster;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;

class GamblerRaster implements StateActionRaster {
  private final GamblerModel gamblerModel;
  private final int offset;

  public GamblerRaster(GamblerModel gamblerModel) {
    this.gamblerModel = gamblerModel;
    offset = (gamblerModel.states().length() - 1) / 2;
  }

  @Override
  public DiscreteModel discreteModel() {
    return gamblerModel;
  }

  @Override
  public Dimension dimensionStateActionRaster() {
    int length = gamblerModel.states().length();
    return new Dimension(length, (length + 1) / 2);
  }

  @Override
  public IntPoint point(Tensor state, Tensor action) {
    return new IntPoint( //
        Scalars.intValueExact((Scalar) state), //
        offset - Scalars.intValueExact((Scalar) action));
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.of(15);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.of(100);
  }

  @Override
  public int joinAlongDimension() {
    return 1;
  }
}
