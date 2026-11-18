// code by jph
package ch.alpine.surich.net.emb;

import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;

@ReflectionMarker
class EmbedDemo extends BaseNetDemo {
  // parameters of supplier are introduced to the gui
  public final Supplier<NetTrainComponent> supplier = new EmbedComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    new EmbedDemo().run();
  }
}
