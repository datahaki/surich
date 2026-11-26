// code by jph
package ch.alpine.surich.net;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;

@ReflectionMarker
public abstract class NetTrainComponentSupplier implements Supplier<NetTrainComponent> {
  public Integer seed = 0;
}
