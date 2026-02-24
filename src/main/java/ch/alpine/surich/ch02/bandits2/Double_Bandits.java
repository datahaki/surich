// code by jph
package ch.alpine.surich.ch02.bandits2;

import ch.alpine.bridge.pro.RunProvider;
import ch.alpine.subare.td.SarsaType;

/** Double Sarsa for maximization bias */
/* package */ enum Double_Bandits implements RunProvider {
  INSTANCE;

  @Override
  public Object runStandalone() {
    BanditsModel banditsModel = new BanditsModel(20);
    BanditsTrain sarsa_Bandits = new BanditsTrain(banditsModel);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 1);
    sarsa_Bandits.handle(SarsaType.EXPECTED, 3);
    sarsa_Bandits.handle(SarsaType.QLEARNING, 2);
    return null;
  }

  static void main() {
    INSTANCE.runStandalone();
  }
}
