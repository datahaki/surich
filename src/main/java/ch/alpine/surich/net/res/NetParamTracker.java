// code by jph
package ch.alpine.surich.net.res;

import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.NetTrainListener;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.TableBuilder;

/** ONLY FOR TESTING ON SMALL NETWORKS */
class NetParamTracker implements NetTrainListener {
  public final TableBuilder tparam = new TableBuilder();

  @Override
  public boolean epoch(NetTrain netTrain) {
    int epoch = netTrain.getTotalEpochs();
    tparam.appendRow(RealScalar.of(epoch), Tensor.of(NetLayers.parameters(netTrain.netModel().netLayer())));
    return true;
  }
}
