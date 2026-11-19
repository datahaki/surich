// code by jph
package ch.alpine.surich.net;

import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import ch.alpine.bridge.cgr.InstanceDiscovery;
import ch.alpine.bridge.cgr.InstanceRecord;
import ch.alpine.subare.net.opt.NetOptimizers;

class BaseNetDemoTest {
  @TestFactory
  Stream<DynamicTest> dynamicTests() throws Exception {
    return InstanceDiscovery.of(getClass().getPackageName(), BaseNetDemo.class).stream() //
        .map(instanceRecord -> DynamicTest.dynamicTest(instanceRecord.toString(), //
            () -> accept(instanceRecord)));
  }

  private static void accept(InstanceRecord<BaseNetDemo> instanceRecord) {
    BaseNetDemo baseNetDemo = instanceRecord.supplier().get();
    for (NetOptimizers netOptimizers : NetOptimizers.values()) {
      NetTrainComponent netTrainComponent = baseNetDemo.supplier().get();
      netTrainComponent.setSize(200, 200);
      OptParam optParam = netTrainComponent.optParam();
      optParam.GD = netOptimizers;
      NetTrainDataBatch netTrainDataBatch = netTrainComponent.createNetTrainDataBatch();
      netTrainDataBatch.netTrain().setMaxEpochs(50);
      netTrainDataBatch.run();
    }
  }
}
