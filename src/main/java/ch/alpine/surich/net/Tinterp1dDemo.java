// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import javax.swing.JComponent;
import javax.swing.JFrame;

import ch.alpine.bridge.awt.WindowClosed;
import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.ListPlot;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.bridge.util.InterruptibleSingleTaskExecutor;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.NetWrap;
import ch.alpine.subare.net.TrainingData;
import ch.alpine.subare.net.lay.ElementwiseLayer;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.opt.NetMomentumGradientDescent;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.run.NetInitialize;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.ext.Serialization;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.qty.Quantity;

@ReflectionMarker
class Tinterp1dDemo implements ManipulateProvider {
  public final Interp1Param interp1Param = new Interp1Param();
  public Integer hidden = 7;
  public Scalar beta = RealScalar.of(0.9);
  public Scalar eta = RealScalar.of(0.03);
  public Integer batch_size = 6;
  public Integer maxEpoch = 1000;
  public Scalar timeout = Quantity.of(1, "s");
  @FieldFuse
  public transient Boolean retrain = false;
  private final InterruptibleSingleTaskExecutor iste = new InterruptibleSingleTaskExecutor();

  class NetLayerWrap {
    final NetLayer netLayer;
    final TrainingData trainingdata;
    final NetTrain netTrain;
    final NetParamStatTracker netParamTracker = new NetParamStatTracker();
    NetLayer copy;
    List<Show> shows;

    public NetLayerWrap(NetLayer netLayer, TrainingData trainingdata) {
      this.netLayer = netLayer;
      this.trainingdata = trainingdata;
      NetInitialize.of(netLayer);
      NetOptimizer netOptimize = new NetMomentumGradientDescent(netLayer, beta, eta);
      netTrain = new NetTrain(netLayer, netOptimize);
      netTrain.addListener(netParamTracker);
      {
        NetTrainListener netTrainListener = new NetTrainListener() {
          @Override
          public boolean epoch(NetTrain netTrain, int epoch) {
            if (epoch % 20 == 0) {
              try {
                copy = Serialization.copy(netTrain.netLayer);
                TrainingData trainingdata = netLayerWrap0.trainingdata;
                Tensor xdata = trainingdata.xdata();
                Tensor ydata = trainingdata.ydata();
                List<Show> list = new ArrayList<Show>();
                {
                  Show show = new Show();
                  show.add(Plot.of(NetWrap.asSUO11(copy), Interp1Param.DOMAIN)).setLabel("gelu");
                  show.add(ListPlot.of(Join.of(1, xdata, ydata)));
                  list.add(show);
                }
                {
                  Show show = new Show();
                  show.setShowLabel("net param");
                  TableBuilder table = netLayerWrap0.netParamTracker.tparam;
                  int n = table.getRow(0).length();
                  for (int i = 1; i < NetParamStatTracker.LABELS.length; ++i) {
                    Showable showable = show.add(ListLinePlot.of(table.getColumns(0, i)));
                    showable.setLabel(NetParamStatTracker.LABELS[i]);
                  }
                  list.add(show);
                }
                {
                  Show show = new Show();
                  show.setShowLabel("loss");
                  show.add(ListLinePlot.of(netLayerWrap0.netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
                  show.add(ListLinePlot.of(netLayerWrap0.netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
                  // show1.add(ListLinePlot.of(netLayerWrap1.netTrain.tloss.getTable())).setLabel("tanh");
                  list.add(show);
                }
                shows = list;
                jComponent.repaint();
              } catch (ClassNotFoundException e) {
                e.printStackTrace();
              } catch (IOException e) {
                e.printStackTrace();
              }
            }
            return true;
          }
        };
        netTrain.addListener(netTrainListener);
        netTrain.addListener(netParamTracker);
      }
    }
  }

  NetLayerWrap netLayerWrap0;
  private final JComponent jComponent = new JComponent() {
    @Override
    protected void paintComponent(Graphics graphics) {
      if (Objects.nonNull(netLayerWrap0) && Objects.nonNull(netLayerWrap0.shows)) {
        Dimension dimension = getSize();
        List<Show> shows = netLayerWrap0.shows;
        // NetLayer copy = netLayerWrap0.copy;
        // IO.println("NET LAYER WRAP");
        // TrainingData trainingdata = netLayerWrap0.trainingdata;
        // Tensor xdata = trainingdata.xdata();
        // Tensor ydata = trainingdata.ydata();
        // {
        shows.get(0).render_autoIndent(graphics, new Rectangle(0, 0, dimension.width / 2, dimension.height / 2));
        shows.get(1).render_autoIndent(graphics, new Rectangle(dimension.width / 2, 0, dimension.width / 2, dimension.height / 2));
        shows.get(2).render_autoIndent(graphics, new Rectangle(0, dimension.height / 2, dimension.width / 2, dimension.height / 2));
        // ShowGridComponent.of(showF, show0, show1);
      }
    };
  };

  @Override
  public Container getContainer() {
    if (retrain) {
      retrain = false;
      // ---
      TrainingData trainingdata = interp1Param.generate();
      netLayerWrap0 = new NetLayerWrap(NetChains.linear( //
          () -> ElementwiseLayer.gelu(), 1, hidden, hidden, hidden, 1), trainingdata);
      iste.submit(() -> netLayerWrap0.netTrain.runAsBatches(trainingdata, batch_size, maxEpoch, timeout));
    }
    return jComponent;
  }

  static void main() {
    Tinterp1dDemo tinterp1dDemo = new Tinterp1dDemo();
    JFrame jFrame = tinterp1dDemo.runStandalone();
    WindowClosed.runs(jFrame, () -> tinterp1dDemo.iste.shutdown());
  }
}
