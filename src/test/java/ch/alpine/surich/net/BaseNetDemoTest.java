// code by jph
package ch.alpine.surich.net;

import java.util.stream.Stream;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

import ch.alpine.bridge.cgr.InstanceDiscovery;
import ch.alpine.bridge.cgr.InstanceRecord;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.los.LossFunction;
import ch.alpine.subare.net.los.NetLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.mod.NetModels;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.chq.DeterminateTensorQ;

class BaseNetDemoTest {
  @TestFactory
  Stream<DynamicTest> batch() throws Exception {
    return InstanceDiscovery.of(getClass().getPackageName(), BaseNetDemo.class).stream() //
        .map(instanceRecord -> DynamicTest.dynamicTest(instanceRecord.toString(), //
            () -> acceptBatch(instanceRecord)));
  }

  private static void acceptBatch(InstanceRecord<BaseNetDemo> instanceRecord) {
    BaseNetDemo baseNetDemo = instanceRecord.supplier().get();
    for (NetOptimizers netOptimizers : NetOptimizers.values()) {
      NetTrainComponent netTrainComponent = baseNetDemo.supplier().get();
      netTrainComponent.setSize(200, 200);
      OptParam optParam = netTrainComponent.optParam();
      optParam.netOptimizers = netOptimizers;
      NetModelDataBatch netTrainDataBatch = netTrainComponent.createNetTrainDataBatch();
      NetTrain netTrain = new NetTrain(netTrainDataBatch.netModel(), optParam.create());
      netTrain.setMaxEpochs(10);
      netTrainDataBatch.init(netTrainComponent.randomGenerator);
      netTrain.runAsBatches(netTrainDataBatch.trainingData(), netTrainDataBatch.batch_size());
    }
  }

  @TestFactory
  Stream<DynamicTest> single() throws Exception {
    return InstanceDiscovery.of(getClass().getPackageName(), BaseNetDemo.class).stream() //
        .map(instanceRecord -> DynamicTest.dynamicTest(instanceRecord.toString(), //
            () -> acceptSingle(instanceRecord)));
  }

  private static void acceptSingle(InstanceRecord<BaseNetDemo> instanceRecord) {
    BaseNetDemo baseNetDemo = instanceRecord.supplier().get();
    for (NetOptimizers netOptimizers : NetOptimizers.values()) {
      NetTrainComponent netTrainComponent = baseNetDemo.supplier().get();
      netTrainComponent.setSize(200, 200);
      OptParam optParam = netTrainComponent.optParam();
      optParam.netOptimizers = netOptimizers;
      NetModelDataBatch netTrainDataBatch = netTrainComponent.createNetTrainDataBatch();
      netTrainDataBatch.init(netTrainComponent.randomGenerator);
      NetModel netModel = netTrainDataBatch.netModel();
      NetTrain netTrain = new NetTrain(netModel, optParam.create());
      NetModels.toString(netModel);
      NetLayer netLayer = netModel.netLayer();
      LossFunction lossFunction = netModel.lossFunction();
      TrainingData trainingData = netTrainDataBatch.trainingData();
      for (int i = 0; i < trainingData.xdata().length(); ++i) {
        Tensor prediction = netLayer.forward(trainingData.xdata().get(i));
        DeterminateTensorQ.require(prediction);
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor target = trainingData.ydata().get(i);
        // IO.println("T="+Dimensions.of(target));
        NetLoss netError = lossFunction.compute(prediction, target);
        // 3. Calculate loss: Apply a loss function to quantify the error.
        /** Loss:
         * A scalar value computed by a loss function (e.g., Mean
         * Squared Error, Cross-Entropy) that aggregates the errors
         * across a sample or batch. It quantifies overall model
         * performance and serves as the objective function to
         * minimize. */
        // 4. Backpropagation: Calculate gradients of the loss with respect to network weights.
        netLayer.back(netError.gradOutput());
        // 5. Weight update: The optimizer adjusts weights to reduce the loss.
        netTrain.netOptimizer().update();
      }
      NetLayers.parametersAsMatrix(netLayer);
    }
  }
}
