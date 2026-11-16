// code by jph
package ch.alpine.surich.net.it1;

import java.util.function.Supplier;

import javax.swing.JFrame;

import ch.alpine.bridge.awt.WindowClosed;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;

@ReflectionMarker
class Interp1Demo extends BaseNetDemo {
  public final Supplier<NetTrainComponent> supplier = new Interp1Component();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    BaseNetDemo baseNetDemo = new Interp1Demo();
    JFrame jFrame = baseNetDemo.runStandalone();
    WindowClosed.runs(jFrame, baseNetDemo::cancel);
  }
}
