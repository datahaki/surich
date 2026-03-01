// code by jph
package ch.alpine.surich;

import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.TestReporter;

import ch.alpine.bridge.cgr.InstanceDiscovery;
import ch.alpine.bridge.pro.RunProvider;
import ch.alpine.bridge.pro.SanityCheckRunProvider;

class RunProviderTest {
  @TestFactory
  Stream<DynamicTest> dynamicTests(TestReporter testReporter) {
    return InstanceDiscovery.of(getClass().getPackageName(), RunProvider.class).stream() //
        .map(instanceRecord -> DynamicTest.dynamicTest(instanceRecord.toString(), //
            () -> {
              testReporter.publishEntry("=="+instanceRecord.toString());
              SanityCheckRunProvider.INSTANCE.accept(instanceRecord);
            }));
  }
}
