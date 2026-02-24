// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch08.maze;

import java.awt.Container;

import ch.alpine.bridge.awt.AwtUtil;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.ReflectionMarker;

@ReflectionMarker
enum AVH_Dynamaze implements ManipulateProvider {
  INSTANCE;

  public AVH_Dynamazes dyna = AVH_Dynamazes.START_0;

  @Override
  public Container getContainer() {
    return AwtUtil.iconAsLabel(dyna.iconImage);
  }

  static void main() throws Exception {
    INSTANCE.runStandalone();
  }
}
