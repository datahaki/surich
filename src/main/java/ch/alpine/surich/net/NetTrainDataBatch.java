// code by jph
package ch.alpine.surich.net;

import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;

public record NetTrainDataBatch(NetTrain netTrain, TrainingData trainingData, int batch_size) {
  public void run() {
    netTrain.runAsBatches(trainingData, batch_size);
  }
}
