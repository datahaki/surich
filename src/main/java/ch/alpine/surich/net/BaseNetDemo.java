// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

import javax.swing.JLabel;
import javax.swing.SwingConstants;

import ch.alpine.bridge.pro.ManipulateProvider;
import ch.alpine.bridge.ref.ann.FieldFuse;
import ch.alpine.bridge.ref.ann.ReflectionMarker;

@ReflectionMarker
public abstract class BaseNetDemo implements ManipulateProvider {
  private final ExecutorService executorService = Executors.newSingleThreadExecutor();
  // ---
  @FieldFuse
  public transient Boolean retrain = false;
  private NetTrainComponent netTrainComponent;

  protected abstract Supplier<NetTrainComponent> supplier();

  @Override
  public Container getContainer() {
    if (retrain) {
      retrain = false;
      if (Objects.isNull(netTrainComponent) || !netTrainComponent.isRunning()) {
        netTrainComponent = supplier().get();
        executorService.submit(() -> {
          try {
            netTrainComponent.runAsBatches();
          } catch (Exception exception) {
            exception.printStackTrace();
          }
        });
      } else {
        netTrainComponent.cancel();
      }
    }
    return Objects.isNull(netTrainComponent) //
        ? new JLabel("press start", SwingConstants.CENTER)
        : netTrainComponent;
  }

  @Override
  public final void windowClosed() {
    if (Objects.nonNull(netTrainComponent))
      netTrainComponent.cancel();
    executorService.shutdown();
  }
}
