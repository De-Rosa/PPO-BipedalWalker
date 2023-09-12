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
    private const int BatchSize = 128;
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
        MonteCarloReturn(trajectory);
        CalculateValueEstimates(trajectory);
        MonteCarloAdvantages(trajectory);
        Normalize(trajectory.Advantages);

        for (int i = 0; i < Epochs; i++)
        {
            List<Batch> batches = CreateBatches(trajectory);
            
            for (int j = 0; j < batches.Count; j++)
            {
                Train(batches[j]);
                Console.WriteLine($"Epoch {i + 1}/{Epochs} | Batch: {j + 1}/{batches.Count}");
            }
        }
        
        //Console.WriteLine($"New standard deviation: {MathF.Exp(_logStandardDeviation - 0.001f)}, previously {Math.Exp(_logStandardDeviation)}");
        //_logStandardDeviation -= 0.001f;
    }

    
    private void Train(Batch batch)
    {
        Matrix advantages = Matrix.ExpandHeight(ListToMatrix(batch.Advantages), _actionSize);
        Matrix returns = ListToMatrix(batch.Returns);
        Matrix actions = ListToMatrix(batch.Actions, _actionSize);
        Matrix states = ListToMatrix(batch.States, _stateSize);
        Matrix oldLogProbabilities = ListToMatrix(batch.LogProbabilities, _actionSize);
        
        _actorNetwork.Zero();
        _criticNetwork.Zero();
        
        Matrix std = GetStandardDeviations(expanded: true);
        // Derivative of the mean squared error
        // 2(V(s) - G)
        // we recalculate the value estimate to get the cache values correct
        Matrix valueEstimate = GetValueEstimateMatrix(states, true);
        Matrix criticLoss = 2 * (valueEstimate - returns);
        
        Matrix mean = GetMeanOutput(states, true);
        Matrix logProbabilities = Matrix.Expand(GetLogProbabilities(mean, std, actions), BatchSize);
        
        // Derivative of L Clip with respect to the policy
        // Equation 20
        Matrix ratio = Matrix.Exponential(logProbabilities - oldLogProbabilities);
        
        Matrix clippedRatio = Matrix.Clip(ratio, 1f + Epsilon, 1f - Epsilon);
        Matrix clippedRatioAdvantage = Matrix.HadamardProduct(clippedRatio, advantages);
        Matrix ratioAdvantage = Matrix.HadamardProduct(ratio, advantages);
        
        Matrix partA = Matrix.Compare(ratioAdvantage, clippedRatioAdvantage, 1f, 0f);
        partA = Matrix.HadamardProduct(partA, advantages);

        Matrix partB = Matrix.CompareNonEquals(clippedRatioAdvantage, ratioAdvantage, 1f, 0f);
        partB = Matrix.HadamardProduct(partB, advantages);

        Matrix partC = Matrix.CompareInRange(ratio, 1f + Epsilon, 1f - Epsilon, 1f, 0f);

        Matrix lClipDerivative = partA + Matrix.HadamardProduct(partB, partC);
        lClipDerivative = Matrix.HadamardDivision(lClipDerivative, Matrix.Exponential(oldLogProbabilities));
        lClipDerivative *= -1f;

        // Derivative of mean with respect to the policy
        // Equation 26
        Matrix probabilities = Matrix.Exponential(logProbabilities);
        
        Matrix actionsMinusMean = actions - mean;
        Matrix variance = Matrix.HadamardProduct(std, std);

        Matrix fraction = Matrix.HadamardDivision(actionsMinusMean, variance);
        Matrix meanDerivative = Matrix.HadamardProduct(probabilities,  fraction);
        
        // Chain rule, dPdMu * dCdP = dCdMu
        Matrix actorLoss = Matrix.HadamardProduct(meanDerivative, lClipDerivative);

        _criticNetwork.FeedBack(criticLoss);
        _actorNetwork.FeedBack(actorLoss);
        
        _criticNetwork.Optimise();
        _actorNetwork.Optimise();
    }

    private Matrix ListToMatrix(float[] list)
    {
        Matrix matrix = Matrix.FromSize(1, BatchSize);
        for (int i = 0; i < BatchSize; i++)
        {
            matrix.SetValue(0, i, list[i]);
        }

        return matrix;
    }

    private Matrix ListToMatrix(Matrix[] list, int matrixSize)
    {
        Matrix matrix = Matrix.FromSize(matrixSize, BatchSize);
        for (int i = 0; i < matrixSize; i++)
        {
            for (int j = 0; j < BatchSize; j++)
            {
                matrix.SetValue(i, j, list[j].GetValue(i, 0));
            }
        }

        return matrix;
    }

    // We calculate V(s) which is called the value function. This is the discounted returns if the AI behaves as expected, and does not take
    // into account the randomness of taking actions
    private float GetValueEstimate(Matrix state, bool cache = false)
    {
        Matrix expandedState = Matrix.Expand(state, BatchSize);
        return _criticNetwork.FeedForward(expandedState, cache).GetValue(0,0);
    }
    
    private Matrix GetValueEstimateMatrix(Matrix state, bool cache = false)
    {
        return _criticNetwork.FeedForward(state, cache);
    }

    private Matrix GetStandardDeviations(bool expanded = false)
    {
        Matrix std = Matrix.FromSize(1, 1);
        std.SetValue(0,0, MathF.Exp(_logStandardDeviation));

        std = Matrix.ExpandHeight(std, _actionSize);
        if (expanded) std = Matrix.Expand(std, BatchSize);

        return std;
    }

    public Matrix SampleActions(Matrix state, out Matrix logProbabilities, out Matrix mean, out Matrix std)
    {
        Matrix expandedState = Matrix.Expand(state, BatchSize);

        mean = GetMeanOutput(expandedState);
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
    private void Normalize(List<float> list)
    {
        if (list.Count == 0) return;
        float mean = list.Average();
        float std = (float) Math.Sqrt(list.Sum(value => Math.Pow(value - mean, 2)) / (list.Count));
        
        for (int i = 0; i < list.Count; i++)
        {
            list[i] -= mean;
            list[i] /= std + 1e-10f;
        }
    }

    private void MonteCarloReturn(Trajectory trajectory)
    {
        trajectory.Returns.Clear();
        
        float discountedReturns = 0;
        for (int i = trajectory.States.Count - 1; i >= 0; i--)
        {
            discountedReturns = trajectory.Rewards[i] + (discountedReturns * Gamma);
            trajectory.Returns.Insert(0, discountedReturns);
        }
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
                batch.Returns[j] = newTrajectory.Returns[value];
                batch.Advantages[j] = newTrajectory.Advantages[value];
                batch.Values[j] = newTrajectory.Values[value];
                
                newTrajectory.States.RemoveAt(value);
                newTrajectory.Actions.RemoveAt(value);
                newTrajectory.LogProbabilities.RemoveAt(value);
                newTrajectory.Rewards.RemoveAt(value);
                newTrajectory.Indexes.RemoveAt(value);
                newTrajectory.Returns.RemoveAt(value);
                newTrajectory.Advantages.RemoveAt(value);
                newTrajectory.Values.RemoveAt(value);
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
