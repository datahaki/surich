// code by jph
package ch.alpine.surich.net.pen;

import java.io.IOException;
import java.nio.file.Path;

import ch.alpine.surich.net.DatasetBuilder;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.io.Import;
import ch.alpine.tensor.qty.Unit;

public enum PenguinDataset {
  INSTANCE;

  public final DatasetBuilder builder;

  private PenguinDataset() {
    Tensor tensor = null;
    try {
      Path path = Unprotect.resourcePath("ch/alpine/surich/net/pen/penguins.csv");
      tensor = Import.of(path);
    } catch (IOException e) {
      e.printStackTrace();
    }
    builder = DatasetBuilder.of(tensor);
    builder.encode(0); // species
    builder.encode(1); // island
    builder.setUnit(2, Unit.of("mm"));
    builder.rescale(2); // bill_length_mm
    builder.setUnit(3, Unit.of("mm"));
    builder.rescale(3); // bill_depth_mm
    builder.setUnit(4, Unit.of("mm"));
    builder.rescale(4); // flipper_length_mm
    builder.setUnit(5, Unit.of("g"));
    builder.rescale(5); // body_mass_g
    builder.encode(6); // sex
    builder.rescale(7); // year
  }

  static void main() {
    // IO.println(INSTANCE.builder);
    // IO.println(MatrixForm.of(INSTANCE.builder.extract(2, 3, 4, 5)));
  }
}
