// code by jph
package ch.alpine.surich.ch05.wireloop;

public enum Wireloops {
  ;
  public static final Wireloop WIRE5_FREES = WireloopHelper.create("wire5", WireloopReward::id_x, WireloopReward.freeSteps());
  public static final Wireloop WIRE5_CONST = WireloopHelper.create("wire5", WireloopReward::id_x, WireloopReward.constantCost());
}
