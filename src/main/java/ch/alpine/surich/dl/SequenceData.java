// code by jph
package ch.alpine.surich.dl;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class SequenceData {
  public static class Sample {
    public NDArray input;
    public NDArray target;

    public Sample(NDArray i, NDArray t) {
      input = i;
      target = t;
    }
  }

  public static List<Sample> generate(NDManager manager, int count, int seqLen) {
    Random rand = new Random();
    List<Sample> data = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      int start = rand.nextInt(20);
      int step = rand.nextInt(4) + 1;
      float[] seq = new float[seqLen];
      for (int j = 0; j < seqLen; j++) {
        seq[j] = start + j * step;
      }
      float next = start + seqLen * step;
      // NDArray input = manager.create(seq).reshape(1, seqLen, 1);
      // NDArray target = manager.create(new float[] { next });
      float mean = 0f;
      for (float v : seq) {
        mean += v;
      }
      mean /= seqLen;
      float variance = 0f;
      for (float v : seq) {
        variance += (v - mean) * (v - mean);
      }
      variance /= seqLen;
      float std = (float) Math.sqrt(variance) + 1e-5f;
      // Normalize sequence
      float[] normalized = new float[seqLen];
      for (int j = 0; j < seqLen; j++) {
        normalized[j] = (seq[j] - mean) / std;
      }
      // ALSO normalize the target using SAME stats
      float normalizedNext = (next - mean) / std;
      NDArray input = manager.create(normalized).reshape(1, seqLen, 1);
      NDArray target = manager.create(new float[] { normalizedNext });
      data.add(new Sample(input, target));
    }
    return data;
  }
}
