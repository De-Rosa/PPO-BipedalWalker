using System;
using System.Collections.Generic;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

public partial class PPO
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetwork;

    private const int DenseSize = 64;
    private const int Epochs = 4;
    private const int BatchSize = 32;
    private const float Gamma = 0.9f; // Discount Factor
    private const float Epsilon = 0.1f; // Clipping Factor
    
    public PPO(int stateSize, int actionSize)
    {
        // Critic neural network transforms a state into a scalar value estimate for use in training the actor
        _criticNetwork = new NeuralNetwork();
        _criticNetwork.AddLayer(new DenseLayer(stateSize, DenseSize));
        _criticNetwork.AddLayer(new LeakyReLULayer());
        _criticNetwork.AddLayer(new DenseLayer(DenseSize, 1));


        // Actor neural network performs an action based on a given state
        _actorNetwork = new NeuralNetwork();
        _actorNetwork.AddLayer(new DenseLayer(stateSize, DenseSize));
        _actorNetwork.AddLayer(new LeakyReLULayer());
        _actorNetwork.AddLayer(new DenseLayer(DenseSize, actionSize));
        _actorNetwork.AddLayer(new TanhLayer());
    }

    public void Train(Trajectory trajectory)
    {
        for (int i = 0; i < Epochs; i++)
        {
            List<Batch> batches = CreateBatches(trajectory);
            foreach (var batch in batches)
            {
                TrainCritic(batch);
                TrainActor(batch);
            }
        }
    }
    
    private void TrainCritic(Batch batch)
    {
        // gradient of mean squared error = -2(G - V(s))
        _criticNetwork.Zero();
        
        // gradient accumulation
        for (int i = 0; i < BatchSize; i++)
        {
            float criticLoss = -2 * (batch.Returns[i] - GetValueEstimate(batch.States[i]));
            criticLoss /= BatchSize;
            _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
        }
        
        _criticNetwork.Optimise();
    }

    
    private void TrainActor(Batch batch)
    {
        _actorNetwork.Zero();

        Matrix actorLoss = Matrix.FromSize(batch.Probabilities[0].GetHeight(), 1);
        
        for (int i = 0; i < BatchSize; i++)
        {
            // L Clip calculation
            Matrix probabilities = GetActionProbabilities(batch.States[i], out Matrix actions);
            Matrix ratio = Matrix.HadamardDivision(probabilities, batch.Probabilities[i]);
            batch.Probabilities[i] = probabilities;

            Matrix ratioAdvantage = ratio * batch.Advantages[i];
            Matrix clippedRatio = Matrix.Clip(ratio, (1 + Epsilon), (1 - Epsilon));
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];

            Matrix minRatio = Matrix.Min(ratioAdvantage, clippedRatioAdvantage);
            actorLoss += minRatio;
        }

        _actorNetwork.FeedBack(-actorLoss);
        _actorNetwork.Optimise();
    }
    
    // We calculate V(s) which is called the value function. This is the discounted returns if the AI behaves as expected, and does not take
    // into account the randomness of taking actions
    private float GetValueEstimate(Matrix state)
    {
        return _criticNetwork.FeedForward(state).GetValue(0,0);
    }
    
    // https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
    // https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
    private Matrix GetActionProbabilities(Matrix state, out Matrix actions)
    {
        actions = _actorNetwork.FeedForward(state);
        
        Matrix probabilities = Matrix.Exponential(actions); 
        float actionSum = probabilities.Sum();
        probabilities /= actionSum;

        return probabilities;
    }

    public int SampleAction(Matrix state, out Matrix actions, out Matrix probabilities)
    {
        probabilities = GetActionProbabilities(state, out actions);
        
        Random random = new Random();
        float number = (float) random.NextDouble();

        float sum = 0;
        
        for (int i = 0; i < probabilities.GetHeight(); i++)
        {
            sum += probabilities.GetValue(i, 0);
            if (number <= sum) return i;
        }

        return 0;
    }

    public void CalculateAdvantages(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            float valueEstimate = GetValueEstimate(trajectory.States[i]);
            trajectory.Advantages.Add(trajectory.Returns[i] - valueEstimate);
        }
    }

    public void GeneralizedAdvantageEstimate(Trajectory trajectory)
    {
        for (int i = trajectory.States.Count - 1; i >= 0; i--)
        {
            
        }
    }

    private List<Batch> CreateBatches(Trajectory trajectory)
    {
        List<Batch> batches = new List<Batch>();
        Random random = new Random();
        Trajectory newTrajectory = trajectory.Copy();
        
        int batchCount = (newTrajectory.States.Count / BatchSize);
        
        for (int i = 0; i < batchCount; i++)
        {
            Batch batch = new Batch(BatchSize);
            
            for (int j = 0; j < BatchSize; j++)
            {
                int index = random.Next(0, newTrajectory.States.Count - 1);
                batch.States[j] = newTrajectory.States[index];
                batch.Rewards[j] = newTrajectory.Rewards[index];
                batch.Returns[j] = newTrajectory.Returns[index];
                batch.Advantages[j] = newTrajectory.Advantages[index];
                batch.Probabilities[j] = newTrajectory.Probabilities[index];
                newTrajectory.States.RemoveAt(index);
            }
            batches.Add(batch);
        }

        return batches;
    }

    // monte-carlo return
    public void RolloutRewards(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            float discountedReturns = 0;
            int discountPower = 0;
            for (int j = i; j < trajectory.Rewards.Count; j++)
            {
                discountedReturns += (float) (trajectory.Rewards[j] * Math.Pow(Gamma, discountPower));
                discountPower += 1;
            }
            trajectory.Returns.Add(discountedReturns);
        }
    }
}