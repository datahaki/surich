// code by jph
package ch.alpine.subare.dl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.transformer.TransformerEncoderBlock;

public class TinyTransformer {
  public static Block buildModel(int seqLen, int dModel) {
    SequentialBlock model = new SequentialBlock();
    // Project scalar input → embedding dimension
    model.add(Linear.builder().setUnits(dModel).build());
    // Transformer encoder (NEW signature)
    model.add(new TransformerEncoderBlock(dModel, // embeddingSize
        4, // headCount
        dModel * 2, // hiddenSize (FFN expansion)
        0.1f, // dropoutProbability
        Activation::relu // activationFunction
    ));
    // Select last token in the sequence
    model.add(new LambdaBlock(ndList -> {
      NDArray x = ndList.singletonOrThrow();
      long last = x.getShape().get(1) - 1;
      return new NDList(x.get(":," + last + ",:"));
    }));
    // Final regression head → predict next value
    model.add(Linear.builder().setUnits(1).build());
    return model;
  }
}
