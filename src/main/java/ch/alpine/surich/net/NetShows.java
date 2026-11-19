// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.grd.AxisMaps;
import ch.alpine.bridge.fig.plt.ListLinePlot;
import ch.alpine.subare.net.NetParamStatTracker;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.sca.Sign;

public enum NetShows {
  ;
  public static Show param(NetParamStatTracker netParamStatTracker) {
    Show show = new Show("parameter stats");
    Tensor[] tensors = netParamStatTracker.tableBuilder().oneToMany();
    for (int i = 0; i < tensors.length; ++i)
      show.add(ListLinePlot.of(tensors[i])).setLabel(NetParamStatTracker.LABELS[i]);
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
