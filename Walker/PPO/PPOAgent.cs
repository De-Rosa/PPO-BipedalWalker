using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Physics.Rendering;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

public class PPOAgent
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetwork;
    private float _logStandardDeviation = -0.5f;

    // Agent hyper-parameters
    private const int DenseSize = 128;
    private const int Epochs = 5;
    private const int BatchSize = 256;
    private const float Gamma = 0.95f; // Discount Factor
    private const float Lambda = 0.95f; // Smoothing Factor
    private const float Epsilon = 0.2f; // Clipping Factor

    private readonly int _stateSize;
    private readonly int _actionSize;

    public PPOAgent(int stateSize, int actionSize)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        
        // Critic neural network transforms a state into a scalar value estimate for use in training the actor
        _criticNetwork = new NeuralNetwork();
        _criticNetwork.AddLayer(new DenseLayer(stateSize, DenseSize, BatchSize));
        _criticNetwork.AddLayer(new LeakyReLULayer());
        _criticNetwork.AddLayer(new DenseLayer(DenseSize, 1, BatchSize));

        // Actor mean neural network transforms a state into an array of means to use in action sampling.
        _actorNetwork = new NeuralNetwork();
        _actorNetwork.AddLayer(new DenseLayer(stateSize, DenseSize, BatchSize));
        _actorNetwork.AddLayer(new LeakyReLULayer());
        _actorNetwork.AddLayer(new DenseLayer(DenseSize, actionSize, BatchSize));
        _actorNetwork.AddLayer(new TanhLayer());
    }

    public void Save(string criticFileLocation, string muFileLocation)
    {
        string[] criticNetwork = _criticNetwork.Save();
        string[] muActorNetwork = _actorNetwork.Save();

        File.WriteAllLines(criticFileLocation, criticNetwork);
        File.WriteAllLines(muFileLocation, muActorNetwork);

    }

    public void Load(string criticFileLocation, string muFileLocation)
    {
        string[] criticNetwork = File.ReadAllLines(criticFileLocation);
        string[] muNetwork = File.ReadAllLines(muFileLocation);

        if (!File.Exists(criticFileLocation) || !File.Exists(muFileLocation))
        {
            Console.WriteLine("Cannot load weights, the weights files do not exist.");
            return;
        }
        
        _criticNetwork.Load(criticNetwork);
        _actorNetwork.Load(muNetwork);
    }

    public void Render(Renderer renderer)
    {
        _actorNetwork.Render(renderer);
    }

    public void Train(Trajectory trajectory)
    {
        for (int i = 0; i < Epochs; i++)
        {
            List<Batch> batches = CreateBatches(trajectory);
            
            for (int j = 0; j < batches.Count; j++)
            {
                CalculateValueEstimates(trajectory);
                MonteCarloReturn(trajectory);
                MonteCarloAdvantages(trajectory);
                UpdateBatches(batches, trajectory);
                
                Train(batches[j], out float valueLoss);
                Console.WriteLine($"Epoch {i + 1}/{Epochs} | Batch: {j + 1}/{batches.Count} | Value loss: {valueLoss}");
            }
        }
        
        //Console.WriteLine($"New standard deviation: {MathF.Exp(_logStandardDeviation - 0.001f)}, previously {Math.Exp(_logStandardDeviation)}");
        //_logStandardDeviation -= 0.001f;
    }

    
    private void Train(Batch batch, out float criticLoss)
    {
        _actorNetwork.Zero();
        _criticNetwork.Zero();

        criticLoss = 0;
        Matrix actorLoss = Matrix.FromZeroes(_actionSize, 1);

        Matrix std = GetStandardDeviations();

        // https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
        // gradient accumulation
        for (int i = 0; i < BatchSize; i++)
        {
            // Derivative of the mean squared error
            // 2(V(s) - G)
            // we recalculate the value estimate to get the cache values correct
            float valueEstimate = GetValueEstimate(batch.States[i], true);
            criticLoss = -2 * (batch.Returns[i] - valueEstimate);
            
            Matrix mean = GetMeanOutput(batch.States[i], true);
            Matrix logProbabilities = GetLogProbabilities(mean, std, batch.Actions[i]);
            
            // Derivative of L Clip with respect to the policy
            // Equation 20
            Matrix ratio = Matrix.Exponential(logProbabilities - batch.LogProbabilities[i]);
            
            Matrix clippedRatio = Matrix.Clip(ratio, 1f + Epsilon, 1f - Epsilon);
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];
            Matrix ratioAdvantage = ratio * batch.Advantages[i];
            
            Matrix partA = Matrix.Compare(ratioAdvantage, clippedRatioAdvantage, 1f, 0f);
            partA *= batch.Advantages[i];

            Matrix partB = Matrix.CompareNonEquals(clippedRatioAdvantage, ratioAdvantage, 1f, 0f);
            partB *= batch.Advantages[i];

            Matrix partC = Matrix.CompareInRange(ratio, 1f + Epsilon, 1f - Epsilon, 1f, 0f);

            Matrix lClipDerivative = partA + Matrix.HadamardProduct(partB, partC);
            lClipDerivative = Matrix.HadamardDivision(lClipDerivative, Matrix.Exponential(batch.LogProbabilities[i]));
            lClipDerivative *= -1f;

            // Derivative of mean with respect to the policy
            // Equation 26
            Matrix probabilities = Matrix.Exponential(logProbabilities);
            
            Matrix actionsMinusMean = batch.Actions[i] - mean;
            Matrix variance = Matrix.HadamardProduct(std, std);

            Matrix fraction = Matrix.HadamardDivision(actionsMinusMean, variance);
            Matrix meanDerivative = Matrix.HadamardProduct(probabilities,  fraction);
            
            // Chain rule, dPdMu * dCdP = dCdMu
            actorLoss += Matrix.HadamardProduct(meanDerivative, lClipDerivative);
        }
        
        criticLoss /= BatchSize;
        actorLoss /= BatchSize;
            
        _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
        _actorNetwork.FeedBack(actorLoss);

        _criticNetwork.Optimise();
        _actorNetwork.Optimise();
    }

    // We calculate V(s) which is called the value function. This is the discounted returns if the AI behaves as expected, and does not take
    // into account the randomness of taking actions
    private float GetValueEstimate(Matrix state, bool cache = false)
    {
        return _criticNetwork.FeedForward(state, cache).GetValue(0,0);
    }

    private Matrix GetStandardDeviations()
    {
        Matrix std = Matrix.FromSize(_actionSize, 1);
        float stdValue = MathF.Exp(_logStandardDeviation);
        
        for (int i = 0; i < _actionSize; i++)
        {
            std.SetValue(i, 0, stdValue);
        }

        return std;
    }

    public Matrix SampleActions(Matrix state, out Matrix logProbabilities, out Matrix mean, out Matrix std)
    {
        mean = GetMeanOutput(state);
        std = GetStandardDeviations();
        
        Matrix actions = Matrix.SampleNormal(mean, std);
        logProbabilities = GetLogProbabilities(mean, std, actions);

        return actions;
    }

    private static Matrix GetLogProbabilities(Matrix mean, Matrix std, Matrix actions)
    {
        return Matrix.LogNormalDensities(mean, std, actions);
    }

    private Matrix GetMeanOutput(Matrix state, bool cache = false)
    {
        return _actorNetwork.FeedForward(state, cache);
    }

    // apparently bootstrapping values does not work practically on single workers
    private void GeneralizedAdvantageEstimate(Trajectory trajectory)
    {
        trajectory.Advantages.Clear();
        trajectory.Returns.Clear();
        
        float nextGae = 0;
        float nextValue = 0;
        
        for (int i = trajectory.States.Count - 1; i >= 0; i--)
        {
            float delta = CalculateDelta(trajectory, i, ref nextValue);
            float GAE = delta + (Gamma * Lambda * nextGae);
            trajectory.Advantages.Add(GAE);
            trajectory.Returns.Add(GAE);
        }
        
        trajectory.Advantages.Reverse();
        trajectory.Returns.Reverse();
    }

    private float CalculateDelta(Trajectory trajectory, int time, ref float nextValue)
    {
        float currentValue = trajectory.Values[time];
        float delta = trajectory.Rewards[time] + (Gamma * nextValue) - currentValue;
        nextValue = currentValue;

        return delta;
    }
    
    private void CalculateValueEstimates(Trajectory trajectory)
    {
        trajectory.Values.Clear();
        
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            float value = GetValueEstimate(trajectory.States[i]);
            trajectory.Values.Add(value);
        }
    }

    // https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
    // https://stackoverflow.com/questions/3141692/standard-deviation-of-generic-list
    private void Standardize(Trajectory trajectory, List<float> list)
    {
        if (list.Count == 0) return;
        float mean = list.Average();
        float std = (float) Math.Sqrt(list.Sum(value => Math.Pow(value - mean, 2)) / (list.Count));
        
        for (int i = 0; i < list.Count; i++)
        {
            list[i] -= mean;
            list[i] /= std;
        }
    }

    private void MonteCarloReturn(Trajectory trajectory)
    {
        trajectory.Returns.Clear();
        
        float discountedReturns = 0;
        for (int i = trajectory.States.Count - 1; i >= 0; i--)
        {
            discountedReturns = trajectory.Rewards[i] + (discountedReturns * Gamma);
            trajectory.Returns.Add(discountedReturns);
        }

        trajectory.Returns.Reverse();
    }

    private void MonteCarloAdvantages(Trajectory trajectory)
    {
        trajectory.Advantages.Clear();
        
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            trajectory.Advantages.Add(trajectory.Returns[i] - trajectory.Values[i]);
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
                int value = random.Next(0, newTrajectory.Indexes.Count);
                int index = newTrajectory.Indexes[value];
                
                batch.Indexes[j] = index;
                batch.States[j] = newTrajectory.States[value];
                batch.Actions[j] = newTrajectory.Actions[value];
                batch.LogProbabilities[j] = newTrajectory.LogProbabilities[value];
                batch.Rewards[j] = newTrajectory.Rewards[value];
                
                newTrajectory.States.RemoveAt(value);
                newTrajectory.Actions.RemoveAt(value);
                newTrajectory.LogProbabilities.RemoveAt(value);
                newTrajectory.Rewards.RemoveAt(value);
                newTrajectory.Indexes.RemoveAt(value);
            }
            
            batches.Add(batch);
        }

        return batches;
    }

    private void UpdateBatches(List<Batch> batches, Trajectory trajectory)
    {
        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.Indexes.Length; i++)
            {
                int index = batch.Indexes[i];
                batch.Values[i] = trajectory.Values[index];
                batch.Returns[i] = trajectory.Returns[index];
                batch.Advantages[i] = trajectory.Advantages[index];
            }
        }
    }
}
