// code by jph
package ch.alpine.surich.net.it2;

import java.util.function.Supplier;

import javax.swing.JFrame;

import ch.alpine.bridge.awt.WindowClosed;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.surich.net.BaseNetDemo;
import ch.alpine.surich.net.NetTrainComponent;

@ReflectionMarker
class SigmoidDemo extends BaseNetDemo {
  public final Supplier<NetTrainComponent> supplier = new SigmoidComponent();

  @Override
  protected Supplier<NetTrainComponent> supplier() {
    return supplier;
  }

  static void main() {
    BaseNetDemo baseNetDemo = new SigmoidDemo();
    JFrame jFrame = baseNetDemo.runStandalone();
    WindowClosed.runs(jFrame, baseNetDemo::cancel);
  }
}
