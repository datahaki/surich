// code by jph
package ch.alpine.surich;

import java.util.Arrays;
import java.util.List;

import ch.alpine.surich.ch03.grid.Ch03Gridworld;

public enum SubareExperimental {
  ;
  static final Object[] OBJECTS = new Object[] { //
      new Ch03Gridworld(), //
  };

  public static <T> List<T> filter(Class<T> cls) {
    List<T> list = Arrays.stream(OBJECTS) //
        .filter(cls::isInstance) //
        .map(cls::cast) //
        .toList();
    if (list.isEmpty())
      throw new IllegalStateException();
    return list;
  }
}
