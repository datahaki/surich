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
  public NetTrain netTrain;

  public NetTrainComponent(int seed) {
    randomGenerator = RandomGeneratorFactory.of("L64X256MixRandom").create(seed);
  }

  public abstract OptParam optParam();

  public abstract NetTrainDataBatch createNetTrainDataBatch();

  public final void runAsBatches() {
    createNetTrainDataBatch().run();
  }

  protected void paintComponent(Graphics graphics, NetModel copy) {
  }

  final int inset = 250;

  @Override
  protected void paintComponent(Graphics graphics) {
    if (Objects.nonNull(copy)) {
      RenderQuality.setQuality(graphics);
      NetShows.layers(graphics, new Point(), copy);
      Dimension dimension = getSize();
      dimension.width -= inset;
      dimension.height /= 2;
      List<Rectangle> list = GraphicsGrid.row(3, new Rectangle(new Point(inset, 0), dimension));
      {
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(0));
        NetShows.pMatrix(copy.netLayer()).render_autoIndent(graphics, list.get(1));
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(2));
      }
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
    Dimension dimension = getSize();
    dimension.height /= 2;
    return GraphicsGrid.row(n, new Rectangle(new Point(0, dimension.height), dimension));
  }

  public final void cancel() {
    running = false;
  }

  public final boolean isRunning() {
    return running;
  }
}
