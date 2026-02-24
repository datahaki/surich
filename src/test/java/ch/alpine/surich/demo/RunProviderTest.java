// code by jph
package ch.alpine.surich.demo;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import ch.alpine.bridge.fig.Show;
import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.pro.RunProvider;
import ch.alpine.bridge.pro.ShowProvider;
import ch.alpine.tensor.ext.ref.InstanceDiscovery;

class RunProviderTest implements Consumer<RunProvider> {
  private static final AtomicInteger COUNT = new AtomicInteger();

  @TestFactory
  Stream<DynamicTest> dynamicTests() {
    return InstanceDiscovery.of("ch.alpine.surich", RunProvider.class).stream() //
        .map(Supplier::get) //
        .map(instance -> DynamicTest.dynamicTest(instance.toString(), () -> accept(instance)));
  }

  @Override
  public void accept(RunProvider runProvider) {
    switch (runProvider) {
    case ManipulateProvider manipulateProvider -> handle(manipulateProvider);
    case ShowProvider showProvider -> handle(showProvider);
    default -> fallback(runProvider);
    }
    COUNT.getAndIncrement();
  }

  public void handle(ManipulateProvider manipulateProvider) {
    Container jComponent = manipulateProvider.getContainer();
    jComponent.setSize(800, 800);
    jComponent.doLayout(); // mandatory
    int width = jComponent.getWidth();
    int height = jComponent.getHeight();
    if (width == 0 || height == 0) {
      throw new IllegalStateException("Component must have a size");
    }
    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = bufferedImage.createGraphics();
    jComponent.printAll(graphics);
    graphics.dispose();
    COUNT.getAndIncrement();
  }

  public void handle(ShowProvider showProvider) {
    Show show = showProvider.getShow();
    Dimension dimension = new Dimension(800, 800);
    BufferedImage bufferedImage = new BufferedImage(dimension.width, dimension.height, BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = bufferedImage.createGraphics();
    show.render_autoIndent(graphics, new Rectangle(dimension));
    graphics.dispose();
  }

  public void fallback(RunProvider runProvider) {
    IO.println(runProvider.getClass().getSimpleName());
  }

  @AfterAll
  static void here() {
    assertTrue(100 <= COUNT.get());
  }
}
