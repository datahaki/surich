// code by jph
package ch.alpine.surich.net;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.err.MeanAbsoluteError;
import ch.alpine.subare.net.err.SquaredErrorLoss;

public enum ErrorType {
  _SquaredErrorLoss(SquaredErrorLoss.INSTANCE),
  _MeanAbsoluteError(MeanAbsoluteError.INSTANCE);

  public final LossFunction lossFunction;

  ErrorType(LossFunction lossFunction) {
    this.lossFunction = lossFunction;
  }
}
