// code by jph
package ch.alpine.surich.net.demo;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;

class FeatureComponentTest {
  @Test
  void test() {
    FeatureComponent featureComponent = new FeatureComponent();
    NetTrainComponent netTrainComponent = featureComponent.get();
    NetTrainDataBatch netTrainDataBatch = netTrainComponent.createNetTrainDataBatch();
    NetTrain netTrain = netTrainDataBatch.netTrain();
    NetLayers.parametersAsMatrix(netTrain.netLayer());
  }
}
