// code by jph
package ch.alpine.surich;

import java.util.Arrays;
import java.util.List;

import ch.alpine.surich.ch03.grid.Ch03Gridworld;
import ch.alpine.surich.ch04.gambler.GamblerModel;
import ch.alpine.surich.ch04.grid.Ch04Gridworld;
import ch.alpine.surich.ch04.rental.CarRental;
import ch.alpine.surich.ch05.blackjack.Blackjack;
import ch.alpine.surich.ch05.infvar.InfiniteVariance;
import ch.alpine.surich.ch05.racetrack.Racetracks;
import ch.alpine.tensor.Rational;

public enum SurichExperimental {
  ;
  static final Object[] OBJECTS = new Object[] { //
      new Ch03Gridworld(), //
      new GamblerModel(20, Rational.of(4, 10)), //
      new Ch04Gridworld(), //
      new CarRental(5), //
      new Blackjack(), //
      new InfiniteVariance(), //
      Racetracks.TRACK0_3, //
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
