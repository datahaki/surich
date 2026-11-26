// code by jph
package ch.alpine.surich.net.demo;

import org.junit.jupiter.api.Test;

import ch.alpine.surich.net.NetModelData;
import ch.alpine.surich.net.NetTrainComponent;

class FeatureComponentTest {
  @Test
  void test() {
    FeatureComponent featureComponent = new FeatureComponent();
    NetTrainComponent netTrainComponent = featureComponent.get();
    NetModelData netTrainDataBatch = netTrainComponent.createNetModelDataBatch();
    // NetTrain netTrain = netTrainDataBatch.netTrain();
    // NetLayers.parametersAsMatrix(netTrain.netLayer());
  }
}
