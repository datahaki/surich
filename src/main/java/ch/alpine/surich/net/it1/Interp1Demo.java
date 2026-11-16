// code by jph
package ch.alpine.surich.net.it1;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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

@ReflectionMarker
class Interp1Demo implements ManipulateProvider {
  private final ExecutorService executorService = Executors.newSingleThreadExecutor();
  public final Interp1Param interp1Param = new Interp1Param();
  //
  public Integer depth = 2;
  public Integer hidden = 7;
  public Scalar beta = RealScalar.of(0.9);
  public Scalar eta = RealScalar.of(0.03);
  public Integer batch_size = 6;
  @FieldFuse
  public transient Boolean retrain = false;
  // ---
  private NetLayerWrap netLayerWrap;

  class NetLayerWrap {
    volatile boolean running = true;
    List<Show> shows = null;

    public void runAsBatches() {
      int[] dims = new int[depth + 2];
      dims[0] = 1;
      dims[dims.length - 1] = 1;
      for (int i = 1; i <= depth; ++i)
        dims[i] = hidden;
      NetLayer netLayer = NetChains.linear(() -> ElementwiseLayer.gelu(), dims);
      NetInitialize.of(netLayer);
      TrainingData trainingData = interp1Param.generate();
      NetOptimizer netOptimize = new NetMomentumGradientDescent(netLayer, beta, eta);
      NetTrain netTrain = new NetTrain(netLayer, netOptimize);
      final NetParamStatTracker netParamTracker = new NetParamStatTracker();
      netTrain.addListener(netParamTracker);
      NetTrainListener netTrainListener = new NetTrainListener() {
        @Override
        public boolean epoch(NetTrain netTrain) {
          int epoch = netTrain.getTotalEpochs();
          if (epoch % 50 == 0) {
            try {
              Tensor xdata = trainingData.xdata();
              Tensor ydata = trainingData.ydata();
              List<Show> list = new ArrayList<Show>();
              {
                Show show = new Show();
                NetLayer copy = Serialization.copy(netTrain.netLayer);
                show.add(Plot.of(NetWrap.asSUO11(copy), Interp1Param.DOMAIN)).setLabel("gelu");
                show.add(ListPlot.of(Join.of(1, xdata, ydata)));
                list.add(show);
              }
              {
                Show show = new Show();
                show.setShowLabel("net param");
                TableBuilder table = netParamTracker.tparam;
                for (int i = 1; i < NetParamStatTracker.LABELS.length; ++i) {
                  Showable showable = show.add(ListLinePlot.of(table.getColumns(0, i)));
                  showable.setLabel(NetParamStatTracker.LABELS[i]);
                }
                list.add(show);
              }
              {
                Show show = new Show();
                show.setShowLabel("loss");
                show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 1))).setLabel("log loss_data");
                show.add(ListLinePlot.of(netTrain.tloss.getColumns(0, 2))).setLabel("log loss_reg");
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
          return running;
        }
      };
      netTrain.addListener(netTrainListener);
      netTrain.runAsBatches(trainingData, batch_size);
    }

    public void cancel() {
      running = false;
    }
  }

  private final JComponent jComponent = new JComponent() {
    @Override
    protected void paintComponent(Graphics graphics) {
      NetLayerWrap _netLayerWrap = netLayerWrap;
      if (Objects.nonNull(_netLayerWrap) && Objects.nonNull(_netLayerWrap.shows)) {
        Dimension dimension = getSize();
        List<Show> shows = _netLayerWrap.shows;
        shows.get(0).render_autoIndent(graphics, new Rectangle(0, 0, dimension.width / 2, dimension.height / 2));
        shows.get(1).render_autoIndent(graphics, new Rectangle(dimension.width / 2, 0, dimension.width / 2, dimension.height / 2));
        shows.get(2).render_autoIndent(graphics, new Rectangle(0, dimension.height / 2, dimension.width / 2, dimension.height / 2));
        // {
        // Show show = new Show();
        // show.add(Plot.of(Exp.FUNCTION, Clips.interval(2, 3)));
        // Rectangle rectangle = new Rectangle(0, dimension.height / 2, dimension.width / 2, dimension.height / 2);
        // rectangle = Show.defaultInsets(rectangle, graphics.getFontMetrics());
        // CoordinateBoundingBox cbb = show.deriveCbb();
        // // IO.println(cbb);
        // ShowableConfig showableConfig = ShowableConfig.log(rectangle, cbb);
        // // IO.println(showableConfig);
        // show.showRender(graphics, showableConfig);
        // }
      }
    };
  };

  @Override
  public Container getContainer() {
    if (retrain) {
      retrain = false;
      if (Objects.isNull(netLayerWrap) || !netLayerWrap.running) {
        netLayerWrap = new NetLayerWrap();
        executorService.submit(netLayerWrap::runAsBatches);
      } else {
        netLayerWrap.cancel();
      }
    }
    return jComponent;
  }

  private void cancel() {
    if (Objects.nonNull(netLayerWrap))
      netLayerWrap.cancel();
    executorService.shutdown();
  }

  static void main() {
    Interp1Demo tinterp1dDemo = new Interp1Demo();
    JFrame jFrame = tinterp1dDemo.runStandalone();
    WindowClosed.runs(jFrame, tinterp1dDemo::cancel);
  }
}
