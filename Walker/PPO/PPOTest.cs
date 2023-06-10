using System;
using System.Collections.Generic;

namespace Physics.Walker.PPO;

public partial class PPOAgent
{
    
    private class TestStep
    {
        public Matrix inputs;
        public float result;
    }
    
    // trains the critic neural network on test batches (approximate summing 5 numbers)
    // used to tweak hyper-parameters and check that the neural network is working
    // will be deleted as final product
    public void TestTrain()
    {
        Random random = new Random();
        float totalLoss = 0;
        
        for (int i = 0; i < Epochs; i++)
        {
            List<TestStep> batch = new List<TestStep>();
            for (int j = 0; j < BatchSize; j++)
            {
                TestStep step = new TestStep();
                step.inputs = Matrix.FromValues(new float[]
                {
                    random.Next(0, 100), random.Next(0, 10), random.Next(0, 10), random.Next(0, 10), random.Next(0, 10)
                });
                step.result = step.inputs.Sum();
                
                batch.Add(step);
            }
            
            TestTrainCritic(batch, out float loss);
            totalLoss += Math.Abs(loss);
        }
        Console.WriteLine($"Total overall loss: {totalLoss}.");
    }

    private void TestTrainCritic(List<TestStep> batch, out float totalLoss)
    {
        _criticNetwork.Zero();

        totalLoss = 0;
        for (int i = 0; i < BatchSize; i++)
        {
            float valueEstimate = GetValueEstimate(batch[i].inputs);
            float criticLoss = -2 * (batch[i].result - valueEstimate);
            
            Console.WriteLine($"Guessed: {valueEstimate}, actual: {batch[i].result}");
            criticLoss /= BatchSize;
            totalLoss += criticLoss;
            _criticNetwork.FeedBack(Matrix.FromValues(new float[][] { new float[] { criticLoss } }));
        }
        Console.WriteLine($"Total Loss: {totalLoss}");
        _criticNetwork.Optimise();
    }
}