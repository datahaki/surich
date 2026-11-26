// code by jph
package ch.alpine.surich.net;

import java.awt.Container;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
  public Integer batch_size = 6;
  public final OptParam optParam = new OptParam();
  private NetTrainComponent netTrainComponent;

  protected abstract NetTrainComponentSupplier supplier();

  @Override
  public Container getContainer() {
    if (retrain) {
      retrain = false;
      if (Objects.isNull(netTrainComponent) || !netTrainComponent.isRunning()) {
        NetTrainComponentSupplier netTrainComponentSupplier = supplier();
        netTrainComponent = netTrainComponentSupplier.get();
        executorService.submit(() -> {
          try {
            netTrainComponent.runAsBatches(optParam, batch_size);
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
