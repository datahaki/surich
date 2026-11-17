// code by jph
package ch.alpine.surich.net;

import java.awt.Graphics;

import javax.swing.JComponent;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.bridge.wdog.SoftWatchdog;
import ch.alpine.bridge.wdog.Watchdog;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
public abstract class NetTrainComponent extends JComponent implements NetTrainListener {
  private final Watchdog watchdog = SoftWatchdog.barking(Quantity.of(50, "ms"));
  private volatile boolean running = true;
  protected NetLayer copy;

  public abstract void runAsBatches();

  @Override
  protected abstract void paintComponent(Graphics graphics);

  @Override
  public final boolean epoch(NetTrain netTrain) {
    if (watchdog.isBarking()) {
      watchdog.notifyWatchdog();
      copy = Serialization.copy(netTrain.netLayer());
      repaint();
    }
    return running;
  }

  public final void cancel() {
    running = false;
  }

  public final boolean isRunning() {
    return running;
  }
}
