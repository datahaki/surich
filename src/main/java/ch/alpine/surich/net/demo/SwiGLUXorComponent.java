// code by jph
package ch.alpine.surich.net.demo;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.List;
import java.util.function.Supplier;

import ch.alpine.bridge.ref.ann.ReflectionMarker;
import ch.alpine.subare.net.NetChain;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.err.BinaryCrossEntropy;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.lay.SigmoidWithLogitsLayer;
import ch.alpine.subare.net.lay.SwiGLULayer;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.subare.net.run.NetTrain;
import ch.alpine.subare.net.run.TrainingData;
import ch.alpine.surich.net.NetShows;
import ch.alpine.surich.net.NetTrainComponent;
import ch.alpine.surich.net.NetTrainDataBatch;
import ch.alpine.surich.net.OptParam;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Slash;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.alg.Tuples;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Mod;

@ReflectionMarker
class SwiGLUXorComponent implements Supplier<NetTrainComponent> {
  public Integer seed = 0;
  public Integer batch_size = 4;
  public Scalar l2 = RealScalar.ZERO;
  public final OptParam optParam = new OptParam();

  @Override
  public NetTrainComponent get() {
    return new InnerComponent();
  }

  public static Scalar xor(Tensor xy) {
    return Mod.function(2).apply(Total.ofVector(xy));
  }

  class InnerComponent extends NetTrainComponent {
    final TrainingData trainingData;
    NetTrain netTrain;

    public InnerComponent() {
      super(seed);
      Tensor X = Tuples.of(Range.of(0, 2), 2);
      Tensor Y = Slash.of(X, SwiGLUXorComponent::xor);
      trainingData = new TrainingData(X, Y);
      IO.println(trainingData);
    }

    @Override
    public OptParam optParam() {
      return optParam;
    };

    @Override
    public NetTrainDataBatch createNetTrainDataBatch() {
      LinearLayer linearLayer1 = new LinearLayer(2, 2);
      linearLayer1.netRegularization = new L2Regularization(l2);
      SwiGLULayer swiGLULayer = new SwiGLULayer(8, 4, 4);
      LinearLayer linearLayer2 = new LinearLayer(2, 1);
      linearLayer2.netRegularization = new L2Regularization(l2);
      swiGLULayer.netRegularization = new L2Regularization(l2);
      SigmoidWithLogitsLayer sigmoidLayer = new SigmoidWithLogitsLayer();
      NetModel netModel = new NetModel(NetChain.join( //
          linearLayer1, //
          ElementwiseLayers.TANH.get(),
          // swiGLULayer, //
          linearLayer2, //
          ScalarLayer.VectorToScalar
      // , //
      // sigmoidLayer
      ), //
          BinaryCrossEntropy.INSTANCE
      // sigmoidLayer.crossEntropyLoss()
      );
      NetLayer netLayer = netModel.netLayer();
      NetLayers.initialize(netLayer, randomGenerator);
      netTrain = new NetTrain(netModel, optParam.create(netLayer));
      netTrain.addListener(netParamStatTracker);
      netTrain.addListener(this);
      return new NetTrainDataBatch(netTrain, trainingData, batch_size);
    }

    @Override
    protected void paintComponent(Graphics graphics, NetModel netModel) {
      NetLayer netLayer = netModel.netLayer();
      List<Rectangle> list = rectangles(4);
      {
        NetShows.param(netParamStatTracker).render_autoIndent(graphics, list.get(1));
        NetShows.loss(netTrain.tloss).render_autoIndent(graphics, list.get(2));
        NetShows.pMatrix(netLayer).render_autoIndent(graphics, list.get(3));
      }
    };
  }
}
