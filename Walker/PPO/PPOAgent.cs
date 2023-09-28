using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Physics.Rendering;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

public class PPOAgent
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetwork;

    private readonly int _stateSize;
    private readonly int _actionSize;
    
    private const string WeightsLocation = "/Users/square/Projects/Physics/Data/Weights/";

    public PPOAgent(int stateSize, int actionSize)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        
        // Critic neural network transforms a state into a scalar value estimate for use in training the actor
        _criticNetwork = new NeuralNetwork();
        // Actor mean neural network transforms a state into an array of means to use in action sampling.
        _actorNetwork = new NeuralNetwork();
        
        CreateNetworks(_criticNetwork, _actorNetwork);
        
        _criticNetwork.Load("critic");
        _actorNetwork.Load("actor");
    }

    private void CreateNetworks(NeuralNetwork critic, NeuralNetwork actor)
    {
        (List<Layer> criticLayers, List<DenseLayer> criticDenseLayers) =
            ParseLayers(Hyperparameters.CriticNeuralNetwork);
        (List<Layer> actorLayers, List<DenseLayer> actorDenseLayers) =
            ParseLayers(Hyperparameters.ActorNeuralNetwork);
        
        critic.AddLayers(criticLayers, criticDenseLayers);
        actor.AddLayers(actorLayers, actorDenseLayers);
    }

    private (List<Layer>, List<DenseLayer>) ParseLayers(string structure)
    {
        string pattern = @"^Input( \|\d+\|| \((ReLU|TanH|LeakyReLU)\))+ Output$";
        bool isValid = Regex.IsMatch(structure, pattern, RegexOptions.IgnoreCase);
        if (!isValid) throw new Exception($"The neural network structure '{structure}' is invalid.");

        string cleanedStructure = Regex.Replace(structure, @"[[|()]|( Output)|(Input )", "");
        string[] tokens = cleanedStructure.Split(" ");

        int denseInput = _stateSize;
        List<Layer> layers = new List<Layer>();
        List<DenseLayer> denseLayers = new List<DenseLayer>();
        
        foreach (var token in tokens)
        {
            Console.WriteLine(token);
            bool isDense = Int32.TryParse(token, out int denseOutput);
            if (isDense)
            {
                DenseLayer layer = new DenseLayer(denseInput, denseOutput);
                layers.Add(layer);
                denseLayers.Add(layer);
                denseInput = denseOutput;
            }

            switch (token)
            {
                case "ReLU":
                    layers.Add(new ReLULayer());
                    break;
                case "LeakyReLU":
                    layers.Add(new LeakyReLULayer());
                    break;
                case "TanH":
                    layers.Add(new TanhLayer());
                    break;
            }
        }
        
        return (layers, denseLayers);
    }

    public void Save()
    {
        string[] criticNetwork = _criticNetwork.Save("critic");
        string[] actorNetwork = _actorNetwork.Save("actor");

        File.WriteAllLines($"{WeightsLocation}{Hyperparameters.CriticWeightFileName}.txt" , criticNetwork);
        File.WriteAllLines($"{WeightsLocation}{Hyperparameters.ActorWeightFileName}.txt" , actorNetwork);

    }

    public void Render(Renderer renderer)
    {
        _actorNetwork.Render(renderer);
    }

    public void Train(Trajectory trajectory, Renderer renderer)
    {
        MonteCarloReturn(trajectory);
        CalculateValueEstimates(trajectory);
        MonteCarloAdvantages(trajectory);
        Normalize(trajectory.Advantages);

        renderer.AddAverageEpisodeReward(trajectory.Rewards.Average());
        
        for (int i = 0; i < Hyperparameters.Epochs; i++)
        {
            List<Batch> batches = CreateBatches(trajectory);
            
            for (int j = 0; j < batches.Count; j++)
            {
                Train(batches[j], out float valueLoss);
                renderer.UpdateConsole(i, j, batches.Count, valueLoss);
            }
        }

        if (Hyperparameters.SaveWeights)
        {
            Save();
        }
        
        //Console.WriteLine($"New standard deviation: {MathF.Exp(_logStandardDeviation - 0.001f)}, previously {Math.Exp(_logStandardDeviation)}");
        //_logStandardDeviation -= 0.001f;
    }

    
    private void Train(Batch batch, out float averageCriticLoss)
    {
        _actorNetwork.Zero();
        _criticNetwork.Zero();
        
        Matrix std = GetStandardDeviations();
        averageCriticLoss = 0;
        
        // https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
        // gradient accumulation
        for (int i = 0; i < Hyperparameters.BatchSize; i++)
        {
            // Derivative of the mean squared error
            // 2(V(s) - G)
            // we recalculate the value estimate to get the cache values correct
            float valueEstimate = GetValueEstimate(batch.States[i], true);
            float criticLoss = 2 * (valueEstimate - batch.Returns[i]);
            
            Matrix mean = GetMeanOutput(batch.States[i], true);
            Matrix logProbabilities = GetLogProbabilities(mean, std, batch.Actions[i]);
            
            // Derivative of L Clip with respect to the policy
            // Equation 20
            Matrix ratio = Matrix.Exponential(logProbabilities - batch.LogProbabilities[i]);
            
            Matrix clippedRatio = Matrix.Clip(ratio, 1f + Hyperparameters.Epsilon, 1f - Hyperparameters.Epsilon);
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];
            Matrix ratioAdvantage = ratio * batch.Advantages[i];
            
            Matrix partA = Matrix.Compare(ratioAdvantage, clippedRatioAdvantage, 1f, 0f);
            partA *= batch.Advantages[i];

            Matrix partB = Matrix.CompareNonEquals(clippedRatioAdvantage, ratioAdvantage, 1f, 0f);
            partB *= batch.Advantages[i];

            Matrix partC = Matrix.CompareInRange(ratio, 1f + Hyperparameters.Epsilon, 1f - Hyperparameters.Epsilon, 1f, 0f);

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
            Matrix actorLoss = Matrix.HadamardProduct(meanDerivative, lClipDerivative);

            criticLoss /= Hyperparameters.BatchSize;
            actorLoss /= Hyperparameters.BatchSize;

            averageCriticLoss += criticLoss;
        
            _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
            _actorNetwork.FeedBack(actorLoss);
        }

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
        float stdValue = MathF.Exp(Hyperparameters.LogStandardDeviation);
        
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
            float GAE = delta + (Hyperparameters.Gamma * Hyperparameters.Lambda * nextGae);
            trajectory.Advantages.Add(GAE);
            trajectory.Returns.Add(GAE);
        }
        
        trajectory.Advantages.Reverse();
        trajectory.Returns.Reverse();
    }

    private float CalculateDelta(Trajectory trajectory, int time, ref float nextValue)
    {
        float currentValue = trajectory.Values[time];
        float delta = trajectory.Rewards[time] + (Hyperparameters.Gamma * nextValue) - currentValue;
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
            discountedReturns = trajectory.Rewards[i] + (discountedReturns * Hyperparameters.Gamma);
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
        
        int batchCount = (newTrajectory.States.Count / Hyperparameters.BatchSize);

        for (int i = 0; i < batchCount; i++)
        {
            Batch batch = new Batch(Hyperparameters.BatchSize);
            
            for (int j = 0; j < Hyperparameters.BatchSize; j++)
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
