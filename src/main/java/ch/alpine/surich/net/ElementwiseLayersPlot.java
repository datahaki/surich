// code by jph
package ch.alpine.surich.net;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.fig.Showable;
import ch.alpine.bridge.fig.plt.Plot;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

class ElementwiseLayersPlot implements ShowProvider {
  @Override
  public Show getShow() {
    Show show = new Show();
    Clip clip = Clips.absolute(2);
    for (var e : ElementwiseLayers.values()) {
      NetLayer netLayer = e.get();
      Showable showable = show.add(Plot.of(s -> (Scalar) netLayer.forward(s), clip));
      showable.setLabel(netLayer.toString());
    }
    return show;
  }

  static void main() {
    new ElementwiseLayersPlot().runStandalone();
  }
}
