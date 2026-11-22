// code by jph
package ch.alpine.surich.net.demo;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;

@ReflectionMarker
class DropoutDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new DropoutComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new DropoutDemo().runStandalone();
  }
}
