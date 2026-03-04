// code by jph
// inspired by Shangtong Zhang
package ch.alpine.surich.ch08.maze;

enum AVH_Dynamazes {
  START_0,
  START_1,
  START_2;

  final Dynamaze dynamaze = DynamazeHelper.create5(ordinal() + 1);
}
