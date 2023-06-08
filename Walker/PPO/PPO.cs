using System;
using System.Collections.Generic;
using System.Linq;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

public partial class PPO
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetwork;
    private NeuralNetwork _oldActorNetwork;

    private const int DenseSize = 64;
    private const int Epochs = 4;
    private const int BatchSize = 128;
    private const float Gamma = 0.9f; // Discount Factor
    private const float Lambda = 0.9f; // Exponential Weight Discount
    private const float Epsilon = 0.2f; // Clipping Factor
    
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
        _actorNetwork.AddLayer(new TanhLayer());
        _actorNetwork.AddLayer(new DenseLayer(DenseSize, actionSize));
        _actorNetwork.AddLayer(new TanhLayer());

        _oldActorNetwork = _actorNetwork;
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
            Matrix newProbabilities = GetActionProbabilities(batch.States[i]);
            Matrix oldProbabilities = GetOldActionProbabilities(batch.States[i]);
            
            Matrix ratio = Matrix.HadamardDivision(newProbabilities, oldProbabilities);

            Matrix ratioAdvantage = ratio * batch.Advantages[i];
            Matrix clippedRatio = Matrix.Clip(ratio, (1 + Epsilon), (1 - Epsilon));
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];

            Matrix minRatio = Matrix.Min(ratioAdvantage, clippedRatioAdvantage);
            actorLoss += minRatio;
        }

        actorLoss /= BatchSize;
        
        _oldActorNetwork = _actorNetwork.Clone();

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
    
    private Matrix GetActionProbabilities(Matrix state)
    {
        Matrix actions = _actorNetwork.FeedForward(state);
        Matrix probabilities = Matrix.Exponential(actions); 
        float actionSum = probabilities.Sum();
        probabilities /= actionSum;

        return probabilities;
    }
    
    private Matrix GetOldActionProbabilities(Matrix state)
    {
        Matrix actions = _oldActorNetwork.FeedForward(state);
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
            float advantage = trajectory.Returns[i] - valueEstimate;
            trajectory.Advantages.Add(advantage);

        }
    }

    // Slow?
    public void GeneralizedAdvantageEstimate(Trajectory trajectory)
    {
        float lambdaGamma = Lambda * Gamma;
        
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            if (i == trajectory.States.Count - 1) break;
            float advantage = 0;
            float decayPower = 0;
            float nextValueEstimate = GetValueEstimate(trajectory.States[i]);
            
            for (int j = i; j < trajectory.States.Count; j++)
            {
                if (j == trajectory.States.Count - 1) break;
                advantage += (float) (CalculateDelta(trajectory, j, ref nextValueEstimate) * Math.Pow(lambdaGamma, decayPower));
                decayPower += 1;
            }
            trajectory.Advantages.Add(advantage);
        }
    }

    private float CalculateDelta(Trajectory trajectory, int time, ref float nextValueEstimate)
    {
        float previousValueEstimate = nextValueEstimate;
        nextValueEstimate = GetValueEstimate(trajectory.States[time + 1]);
        float delta = trajectory.Rewards[time] + (Gamma * nextValueEstimate) - previousValueEstimate;
        return delta;
    }

    public void MonteCarloReturn(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            float discountedReturns = 0;
            int decayPower = 0;
            for (int j = i; j < trajectory.Rewards.Count; j++)
            {
                discountedReturns += (float) (trajectory.Rewards[j] * Math.Pow(Gamma, decayPower));
                decayPower += 1;
            }
            trajectory.Returns.Add(discountedReturns);
        }
    }

    // https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
    // https://stackoverflow.com/questions/3141692/standard-deviation-of-generic-list
    public void NormaliseAdvantages(Trajectory trajectory)
    {
        float mean = trajectory.Advantages.Average();
        float std = (float) Math.Sqrt(trajectory.Advantages.Average(v => Math.Pow(v - mean, 2)));
        
        for (int i = 0; i < trajectory.Advantages.Count; i++)
        {
            trajectory.Advantages[i] -= mean;
            trajectory.Advantages[i] /= std;
        }
    }

    public void RemoveTerminalState(Trajectory trajectory)
    {
        trajectory.Actions.RemoveAt(trajectory.Actions.Count - 1);
        trajectory.States.RemoveAt(trajectory.States.Count - 1);
        trajectory.Rewards.RemoveAt(trajectory.Rewards.Count - 1);
        trajectory.Returns.RemoveAt(trajectory.Returns.Count - 1);
        trajectory.Probabilities.RemoveAt(trajectory.Probabilities.Count - 1);
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
}