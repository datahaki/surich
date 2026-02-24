// code by jph
package ch.alpine.subare.dl;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;

public class TrainExample {
  public static void main(String[] args) throws Exception {
    try (NDManager manager = NDManager.newBaseManager()) {
      int seqLen = 5;
      var data = SequenceData.generate(manager, 500, seqLen);
      Model model = Model.newInstance("toy-transformer");
      model.setBlock(TinyTransformer.buildModel(seqLen, 32));
      Loss loss = Loss.l2Loss();
      Trainer trainer = model.newTrainer(new DefaultTrainingConfig(loss).optOptimizer(Adam.builder().build()));
      trainer.initialize(new Shape(1, seqLen, 1));
      for (int epoch = 0; epoch < 20; epoch++) {
        float totalLoss = 0f;
        for (var sample : data) {
          try (GradientCollector gc = trainer.newGradientCollector()) {
            NDArray pred = trainer.forward(new NDList(sample.input)).singletonOrThrow();
            NDArray l = loss.evaluate(new NDList(sample.target), new NDList(pred));
            totalLoss += l.getFloat();
            gc.backward(l);
          }
          trainer.step();
        }
        System.out.println("Epoch " + epoch + " loss = " + totalLoss);
      }
      // Test prediction
      var test = SequenceData.generate(manager, 1, seqLen).get(0);
      NDArray prediction = trainer.forward(new NDList(test.input)).singletonOrThrow();
      System.out.println("Input: " + test.input);
      System.out.println("True next: " + test.target);
      System.out.println("Predicted: " + prediction);
    }
  }
}
