// code by jph
package ch.alpine.surich.net;

import java.util.random.RandomGenerator;

import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;

public record NetTrainDataBatch(NetTrain netTrain, TrainingData trainingData, int batch_size) {
  public void init(RandomGenerator randomGenerator) {
    NetLayers.initialize(netTrain.netModel().netLayer(), randomGenerator);
  }

  public void run() {
    netTrain.runAsBatches(trainingData, batch_size);
  }
}
