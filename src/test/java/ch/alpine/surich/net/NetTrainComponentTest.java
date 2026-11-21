package ch.alpine.surich.net;

import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import ch.alpine.bridge.cgr.InstanceDiscovery;
import ch.alpine.bridge.cgr.InstanceRecord;
import ch.alpine.subare.net.NetExtract;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.run.NetTrain;

/* point_wise
 * sequential */
class NetTrainComponentTest {
  @TestFactory
  Stream<DynamicTest> batch() throws Exception {
    return InstanceDiscovery.of(getClass().getPackageName(), BaseNetDemo.class).stream() //
        .map(instanceRecord -> DynamicTest.dynamicTest(instanceRecord.toString(), //
            () -> accept(instanceRecord)));
  }

  private static void accept(InstanceRecord<BaseNetDemo> instanceRecord) {
    BaseNetDemo baseNetDemo = instanceRecord.supplier().get();
    NetTrainComponent netTrainComponent = baseNetDemo.supplier().get();
    netTrainComponent.setSize(200, 200);
    NetTrainDataBatch netTrainDataBatch = netTrainComponent.createNetTrainDataBatch();
    NetTrain netTrain = netTrainDataBatch.netTrain();
    NetLayer netLayer = netTrain.netLayer();
    IO.println(instanceRecord.friendly() + "================");
    List<String> list = NetExtract.toString(netLayer);
    list.forEach(IO::println);
  }
}
