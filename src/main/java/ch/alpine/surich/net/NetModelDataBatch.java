// code by jph
package ch.alpine.surich.net;

import java.util.random.RandomGenerator;

import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.run.TrainingData;

public record NetModelDataBatch(NetModel netModel, TrainingData trainingData, int batch_size) {
  public void init(RandomGenerator randomGenerator) {
    NetLayers.initialize(netModel.netLayer(), randomGenerator);
  }
}
