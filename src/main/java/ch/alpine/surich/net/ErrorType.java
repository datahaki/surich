// code by jph
package ch.alpine.surich.net;

import ch.alpine.subare.net.los.LossFunction;
import ch.alpine.subare.net.los.MeanAbsoluteError;
import ch.alpine.subare.net.los.SquaredErrorLoss;

public enum ErrorType {
  _SquaredErrorLoss(SquaredErrorLoss.INSTANCE),
  _MeanAbsoluteError(MeanAbsoluteError.INSTANCE);

  public final LossFunction lossFunction;

  ErrorType(LossFunction lossFunction) {
    this.lossFunction = lossFunction;
  }
}
