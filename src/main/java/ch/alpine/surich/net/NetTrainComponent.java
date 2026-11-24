// code by jph
package ch.alpine.surich.net;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import javax.swing.JComponent;

import ch.alpine.bridge.awt.RenderQuality;
import ch.alpine.bridge.fig.GraphicsGrid;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.bridge.wdog.SoftWatchdog;
import ch.alpine.bridge.wdog.Watchdog;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.run.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
public abstract class NetTrainComponent extends JComponent implements NetTrainListener {
  private final Watchdog watchdog = SoftWatchdog.barking(Quantity.of(50, "ms"));
  protected final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
  private volatile boolean running = true;
  private NetModel copy;
  protected final RandomGenerator randomGenerator;

  public NetTrainComponent(int seed) {
    randomGenerator = RandomGeneratorFactory.of("L64X256MixRandom").create(seed);
  }

  public abstract OptParam optParam();

  public abstract NetTrainDataBatch createNetTrainDataBatch();

  public final void runAsBatches() {
    createNetTrainDataBatch().run();
  }

  protected abstract void paintComponent(Graphics graphics, NetModel copy);

  @Override
  protected void paintComponent(Graphics graphics) {
    if (Objects.nonNull(copy)) {
      RenderQuality.setQuality(graphics);
      NetShows.layers(graphics, new Point(), copy);
      paintComponent(graphics, copy);
    }
  }

  @Override
  public final boolean epoch(NetTrain netTrain) {
    if (watchdog.isBarking()) {
      watchdog.notifyWatchdog();
      copy = Serialization.copy(netTrain.netModel());
      repaint();
    }
    return running;
  }

  public List<Rectangle> rectangles(int n) {
    int inset = 250;
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
