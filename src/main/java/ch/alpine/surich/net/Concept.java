// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

import javax.swing.JComponent;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.ElementwiseLayer;
import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetTrain;
import ch.alpine.subare.net.NetTrainListener;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

@ReflectionMarker
class Concept implements ManipulateProvider, NetTrainListener {
  public final Interp1Param interp1Param = new Interp1Param();
  public Integer hidden = 7;
  public Scalar learningRate = RealScalar.of(0.2);
  public Integer batch_size = 6;
  public Integer max_epochs = 1000;
  public Scalar timeout = Quantity.of(1, "s");
  @FieldFuse
  public transient Boolean train = false;
  private int count = 0;
  private TrainingData trainingdata;
  private NetTrain netTrainCopy;
  private Thread thread;
  private final JComponent jComponent = new JComponent() {
    @Override
    protected void paintComponent(Graphics _g) {
      _g.drawString("" + count, 10, 10);
      NetTrain netTrain = netTrainCopy;
      if (Objects.nonNull(netTrain)) {
        Dimension dimension = getSize();
        int width = dimension.width;
        int height = dimension.height / 2;
        {
          Show show = new Show();
          Clip domain = Clips.absoluteOne();
          show.setShowLabel("function");
          show.add(Plot.of(NetLayers.asSUO11(netTrain.netLayer), domain)).setLabel("gelu");
          show.add(ListPlot.of(Join.of(1, trainingdata.xdata(), trainingdata.ydata())));
          show.render_autoIndent(_g, new Rectangle(0, 0, width, height));
        }
        {
          Show show = new Show();
          show.setShowLabel("loss");
          show.add(ListLinePlot.of(netTrain.tloss.getTable())).setLabel("gelu");
          show.render_autoIndent(_g, new Rectangle(0, height, width, height));
        }
      }
    }
  };

  @Override
  public boolean epoch(NetTrain netTrain, int epoch) {
    if (epoch % 50 == 0) {
      count = epoch;
      try {
        // TODO unacceptable to copy xdata and ydata!
        netTrainCopy = Serialization.copy(netTrain);
      } catch (Exception e) {
        e.printStackTrace();
      }
      jComponent.repaint();
    }
    return true;
  }

  @Override
  public Container getContainer() {
    if (train) {
      train = false;
      trainingdata = interp1Param.generate();
      // ---
      NetLayer netLayer = NetChains.linear( //
          () -> ElementwiseLayer.gelu(), 1, hidden, hidden, hidden, 1);
      NetInitialize.of(netLayer);
      NetTrain netTrain = new NetTrain(netLayer, trainingdata);
      netTrain.addListener(this);
      thread = new Thread(() -> {
        netTrain.runAsBatches(learningRate, batch_size, max_epochs, timeout);
      });
      thread.start();
      count++;
    }
    return jComponent;
  }

  static void main() {
    new Concept().runStandalone();
  }
}
