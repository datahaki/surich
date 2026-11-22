// code by jph
package ch.alpine.surich.net;

import java.awt.Graphics;
import java.awt.Point;
import java.util.List;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.grd.AxisMaps;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.bridge.fig.plt.MatrixPlot;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.run.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.col.ColorDataGradients;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.sca.Sign;

public enum NetShows {
  ;
  public static void layers(Graphics graphics, Point point, NetLayer netLayer) {
    List<String> list = NetLayers.toString(netLayer);
    int piy = 0;
    for (String string : list) {
      piy += 16;
      graphics.drawString(string, point.x, point.y + piy);
    }
  }

  public static Show param(NetParamStatTracker netParamStatTracker) {
    Show show = new Show("parameter stats");
    Tensor[] tensors = netParamStatTracker.tableBuilder().oneToMany();
    for (int i = 0; i < tensors.length; ++i)
      show.add(ListLinePlot.of(tensors[i])).setLabel(NetParamStatTracker.LABELS[i]);
    return show;
  }

  public static Show pMatrix(NetLayer netLayer) {
    Show show = new Show("parameters");
    Tensor matrix = NetLayers.parametersAsMatrix(netLayer);
    show.add(MatrixPlot.of(matrix, ColorDataGradients.THERMOMETER, true));
    return show;
  }

  public static Show loss(TableBuilder tloss) {
    Show show = new Show("losses");
    Tensor[] tensors = tloss.oneToMany();
    for (int i = 0; i < tensors.length; ++i) {
      Tensor XY = Tensor.of(tensors[i].stream().filter(xy -> Sign.isPositive(xy.Get(1))));
      if (!Tensors.isEmpty(XY))
        show.add(ListLinePlot.of(XY)).setLabel(NetTrain.LOSS_LABELS[i]);
    }
    show.showOptions.axisOptionsY.axisMap = AxisMaps.LOG10;
    return show;
  }
}
