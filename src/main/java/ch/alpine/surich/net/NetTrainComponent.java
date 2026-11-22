// code by jph
package ch.alpine.surich.net;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;

import javax.swing.JComponent;

import ch.alpine.bridge.fig.GraphicsGrid;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.bridge.wdog.SoftWatchdog;
import ch.alpine.bridge.wdog.Watchdog;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
public abstract class NetTrainComponent extends JComponent implements NetTrainListener {
  private final Watchdog watchdog = SoftWatchdog.barking(Quantity.of(50, "ms"));
  private volatile boolean running = true;
  protected NetLayer copy;

  public abstract OptParam optParam();

  public abstract NetTrainDataBatch createNetTrainDataBatch();

  public final void runAsBatches() {
    createNetTrainDataBatch().run();
  }

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

  public List<Rectangle> rectangles(int n) {
    int inset = 200;
    Dimension dimension = getSize();
    dimension.width -= inset;
    return GraphicsGrid.rectangles(n, new Rectangle(new Point(inset, 0), dimension));
  }

  public final void cancel() {
    running = false;
  }

  public final boolean isRunning() {
    return running;
  }
}
