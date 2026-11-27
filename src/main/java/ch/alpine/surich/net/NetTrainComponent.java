// code by jph
package ch.alpine.surich.net;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.util.ArrayList;
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
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.run.NetParamDeltaTracker;
import ch.alpine.subare.net.run.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
public abstract class NetTrainComponent extends JComponent implements NetTrainListener {
  private final Watchdog watchdog = SoftWatchdog.barking(Quantity.of(50, "ms"));
  private final NetParamStatTracker netParamStatTracker = new NetParamStatTracker();
  private final NetParamDeltaTracker netParamDeltaTracker = new NetParamDeltaTracker();
  private volatile boolean running = true;
  private NetTrain netTrain;
  private NetModel ferry;
  protected final RandomGenerator randomGenerator;
  protected final List<String> notes = new ArrayList<>();

  public NetTrainComponent(int seed) {
    randomGenerator = RandomGeneratorFactory.of("L64X256MixRandom").create(seed);
  }

  public abstract NetModelData createNetModelDataBatch();

  public final void runAsBatches(OptParam optParam, int batch_size) {
    NetModelData netTrainDataBatch = createNetModelDataBatch();
    NetModel netModel = netTrainDataBatch.netModel();
    NetLayers.initialize(netModel.netLayer(), randomGenerator);
    netTrain = new NetTrain(netModel, optParam.create());
    netTrain.addListener(netParamStatTracker);
    netTrain.addListener(netParamDeltaTracker);
    netTrain.addListener(this);
    netTrain.setMaxEpochs(20000);
    netTrain.runAsBatches(netTrainDataBatch.trainingData(), batch_size);
  }

  protected void paintComponent(Graphics graphics, NetModel copy) {
  }

  final int inset = 300;

  @Override
  protected final void paintComponent(Graphics graphics) {
    NetModel copy = ferry;
    if (Objects.nonNull(copy)) {
      RenderQuality.setQuality(graphics);
      {
        int piy = NetShows.layers(graphics, new Point(), copy);
        piy += 16;
        for (String s : notes) {
          graphics.drawString(s, 0, piy);
          piy += 16;
        }
      }
      Dimension dimension = getSize();
      dimension.width -= inset;
      dimension.height /= 2;
      List<Rectangle> list = GraphicsGrid.row(3, new Rectangle(new Point(inset, 0), dimension));
      {
        try {
          NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(0));
        } catch (Exception e) {
          IO.println(netTrain.tloss.getTable());
        }
        NetShows.pMatrix(copy.netLayer()).render_autoIndent(graphics, list.get(1));
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(2));
      }
      paintComponent(graphics, copy);
    }
  }

  @Override
  public final boolean epoch(NetTrain netTrain) {
    // dont draw more often than per epoch
    if (watchdog.isBarking()) {
      watchdog.notifyWatchdog();
      ferry = Serialization.copy(netTrain.netModel());
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
