// code by jph
package ch.alpine.surich.net;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Drop;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.io.MatrixForm;
import ch.alpine.tensor.io.StringTensor;
import ch.alpine.tensor.mat.MatrixQ;
import ch.alpine.tensor.qty.Quantity;
import ch.alpine.tensor.qty.Unit;

public class DatasetBuilder {
  public static DatasetBuilder of(Tensor matrix) {
    MatrixQ.require(matrix);
    String[] strings = matrix.get(0).stream().map(Tensor::toString).toArray(String[]::new);
    return new DatasetBuilder(strings, Drop.head(matrix, 1));
  }

  private final String[] header;
  private final Tensor matrix;

  private DatasetBuilder(String[] header, Tensor matrix) {
    this.header = header;
    this.matrix = matrix;
  }

  public void setUnit(int col, Unit unit) {
    matrix.set(s -> Quantity.of((Scalar) s, unit), Tensor.ALL, col);
  }

  public int encode(int col) {
    Int myI = new Int();
    Map<Tensor, Scalar> map = new HashMap<>();
    matrix.set( //
        label -> map.computeIfAbsent(label, _ -> RealScalar.of(myI.getAndIncrement())), //
        Tensor.ALL, col);
    return map.size();
  }

  public int distinctCount(int col) {
    return (int) matrix.get(Tensor.ALL, col).stream().distinct().count();
  }

  public void rescale(int col) {
    Rescale rescale = new Rescale(matrix.get(Tensor.ALL, col));
    matrix.set(rescale.result(), Tensor.ALL, col);
  }

  public Tensor extract(int... cols) {
    return Tensor.of(matrix.stream() //
        .map(row -> Tensor.of(IntStream.of(cols) //
            .mapToObj(row::Get), cols.length)), //
        matrix.length());
  }

  public Tensor single(int col) {
    return matrix.get(Tensor.ALL, col);
  }

  @Override
  public String toString() {
    return MatrixForm.of(Join.of( //
        Tensors.of(StringTensor.vector(header)), //
        matrix));
  }
}
