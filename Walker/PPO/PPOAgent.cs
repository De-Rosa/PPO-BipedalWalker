using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using Physics.Rendering;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

public partial class PPOAgent
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetworkMean;

    private const int DenseSize = 32;
    private const int Epochs = 10;
    private const int BatchSize = 256;
    private const float Gamma = 0.99f; // Discount Factor
    private const float Lambda = 0.95f; // Smoothing Factor
    private const float Epsilon = 0.2f; // Clipping Factor
    private const float StandardDeviation = 0.6f; // Non-learnable, can be learned.

    public PPOAgent(int stateSize, int actionSize)
    {
        // Critic neural network transforms a state into a scalar value estimate for use in training the actor
        _criticNetwork = new NeuralNetwork();
        _criticNetwork.AddLayer(new DenseLayer(stateSize, DenseSize));
        _criticNetwork.AddLayer(new LeakyReLULayer());
        _criticNetwork.AddLayer(new DenseLayer(DenseSize, 1));


        // Actor mean neural network calculates a mean for a given action.
        _actorNetworkMean = new NeuralNetwork();
        _actorNetworkMean.AddLayer(new DenseLayer(stateSize, DenseSize));
        _actorNetworkMean.AddLayer(new LeakyReLULayer());
        _actorNetworkMean.AddLayer(new DenseLayer(DenseSize, actionSize));
        _actorNetworkMean.AddLayer(new TanhLayer());
        
    }

    public void Save(string criticFileLocation, string actorFileLocation)
    {
        string[] criticNetwork = _criticNetwork.Save();
        string[] actorNetwork = _actorNetworkMean.Save();

        File.WriteAllLines(criticFileLocation, criticNetwork);
        File.WriteAllLines(actorFileLocation, actorNetwork);
    }

    public void Load(string criticFileLocation, string actorFileLocation)
    {
        string[] criticNetwork = File.ReadAllLines(criticFileLocation);
        string[] actorNetwork = File.ReadAllLines(actorFileLocation);
        
        if (!File.Exists(criticFileLocation) || !File.Exists(actorFileLocation))
        {
            Console.WriteLine("Cannot load weights, the weights files do not exist.");
            return;
        }
        
        _criticNetwork.Load(criticNetwork);
        _actorNetworkMean.Load(actorNetwork);
    }

    public void Render(Renderer renderer)
    {
        _actorNetworkMean.Render(renderer);
    }

    public void Train(Trajectory trajectory)
    {
        CalculateValueEstimates(trajectory);
        MonteCarloReturn(trajectory);
        GeneralizedAdvantageEstimate(trajectory);
        
        for (int i = 0; i < Epochs; i++)
        {
            List<Batch> batches = CreateBatches(trajectory);
            for (int j = 0; j < batches.Count; j++)
            {
                Train(batches[j], out float valueLoss, out float lClip);
                Console.WriteLine($"Epoch {i + 1}/{Epochs} | Batch: {j + 1}/{batches.Count} | Value loss: {valueLoss} | LClip: {lClip}");
            }
        }
    }

    private void Train(Batch batch, out float valueLoss, out float lClip)
    {
        _actorNetworkMean.Zero();
        _criticNetwork.Zero();

        valueLoss = 0;
        lClip = 0;
        
        // https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
        // gradient accumulation
        for (int i = 0; i < BatchSize; i++)
        {
            // Derivative of the mean squared error
            // -2(G - V(s))
            float criticLoss = -2 * (batch.Returns[i] - GetValueEstimate(batch.States[i]));
            criticLoss /= BatchSize;
            valueLoss += criticLoss;
            
            SampleActions(batch.States[i], out Matrix logProbabilities, out Matrix mean, out Matrix std);
            
            // Derivative of L Clip with respect to the policy
            // Equation 20
            Matrix ratio = Matrix.Exponential(logProbabilities - batch.LogProbabilities[i]);
            Matrix clippedRatio = Matrix.Clip(ratio, 1 + Epsilon, 1 - Epsilon);
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];
            Matrix ratioAdvantage = ratio * batch.Advantages[i];

            Matrix lClipMatrix = Matrix.Min(ratioAdvantage, clippedRatioAdvantage);
            lClip = lClipMatrix.Mean();

            Matrix partA = Matrix.Compare(ratioAdvantage, clippedRatioAdvantage, 1, 0);
            partA *= batch.Advantages[i];

            Matrix partB = Matrix.CompareNonEquals(clippedRatioAdvantage, ratioAdvantage, 1, 0);
            partB *= batch.Advantages[i];

            Matrix partC = Matrix.CompareInRange(ratio, 1 + Epsilon, 1 - Epsilon, 1, 0);

            Matrix lClipDerivative = partA + Matrix.HadamardProduct(partB, partC);
            lClipDerivative *= -1;
            lClipDerivative = Matrix.HadamardDivision(lClipDerivative, Matrix.Exponential(batch.LogProbabilities[i]));

            lClipDerivative -= Matrix.NormalEntropies(std);
            
            // Derivative of mean with respect to the policy
            // Equation 26
            Matrix probabilities = Matrix.Exponential(logProbabilities);
            
            Matrix actionsMinusMean = batch.Actions[i] - mean;
            Matrix variance = Matrix.HadamardProduct(std, std);

            Matrix meanDerivative = Matrix.HadamardProduct(probabilities,  (Matrix.HadamardDivision(actionsMinusMean, variance + 1e-9f)));
            meanDerivative = Matrix.HadamardProduct(meanDerivative, lClipDerivative);
            
            meanDerivative /= BatchSize;
            
            _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
            _actorNetworkMean.FeedBack(meanDerivative);

        }
        
        _criticNetwork.Optimise();
        _actorNetworkMean.Optimise();
    }

    // We calculate V(s) which is called the value function. This is the discounted returns if the AI behaves as expected, and does not take
    // into account the randomness of taking actions
    private float GetValueEstimate(Matrix state)
    {
        return _criticNetwork.FeedForward(state).GetValue(0,0);
    }

    public Matrix SampleActions(Matrix state, out Matrix logProbabilities, out Matrix mean, out Matrix std)
    {
        mean = GetMeanOutput(state);
        std = GetStdMatrix(mean.GetHeight());

        Matrix actions = Matrix.SampleNormal(mean, std);
        actions = Matrix.Clip(actions, 1, -1);

        logProbabilities = GetLogProbability(mean, std, actions);

        return actions;
    }

    private Matrix GetStdMatrix(int height)
    {
        Matrix std = Matrix.FromSize(height, 1);
        
        for (int i = 0; i < height; i++)
        {
            std.SetValue(i, 0, StandardDeviation);
        }

        return std;
    }

    private static Matrix GetLogProbability(Matrix mean, Matrix std, Matrix actions)
    {
        return Matrix.NormalDensities(mean, std, actions);
    }

    private Matrix GetMeanOutput(Matrix state)
    {
        return _actorNetworkMean.FeedForward(state);
    }

    private void GeneralizedAdvantageEstimate(Trajectory trajectory)
    {
        float lambdaGamma = Lambda * Gamma;
        
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            if (i == trajectory.States.Count - 1) break;
            float advantage = 0;
            float decayPower = 0;
            float nextValueEstimate = trajectory.Values[i];
            
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
        nextValueEstimate = trajectory.Values[time + 1];
        float delta = trajectory.Rewards[time] + (Gamma * nextValueEstimate) - previousValueEstimate;
        return delta;
    }

    private void CalculateReturns(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.Advantages.Count; i++)
        {
            float value = trajectory.Advantages[i] + trajectory.Values[i];
            trajectory.Returns.Add(value);
        }
    }

    private void CalculateValueEstimates(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            float value = GetValueEstimate(trajectory.States[i]);
            trajectory.Values.Add(value);
        }
    }

    // https://datascience.stackexchange.com/questions/20098/why-do-we-normalize-the-discounted-rewards-when-doing-policy-gradient-reinforcem
    // https://stackoverflow.com/questions/3141692/standard-deviation-of-generic-list
    private void StandardizeAdvantages(Trajectory trajectory)
    {
        float mean = trajectory.Advantages.Average();
        float std = (float) Math.Sqrt(trajectory.Advantages.Sum(value => Math.Pow(value - mean, 2)) / (trajectory.Advantages.Count - 1));
        
        for (int i = 0; i < trajectory.Advantages.Count; i++)
        {
            trajectory.Advantages[i] -= mean;
            trajectory.Advantages[i] /= std;
        }
    }

    private void MonteCarloReturn(Trajectory trajectory)
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

    private void CalculateAdvantages(Trajectory trajectory)
    {
        for (int i = 0; i < trajectory.Returns.Count; i++)
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
                int index = random.Next(0, newTrajectory.States.Count - 1);
                batch.States[j] = newTrajectory.States[index];
                batch.Actions[j] = newTrajectory.Actions[index];
                batch.Means[j] = newTrajectory.Means[index];
                batch.Stds[j] = newTrajectory.Stds[index];
                batch.LogProbabilities[j] = newTrajectory.LogProbabilities[index];
                batch.Rewards[j] = newTrajectory.Rewards[index];
                batch.Values[j] = newTrajectory.Values[index];
                batch.Returns[j] = newTrajectory.Returns[index];
                batch.Advantages[j] = newTrajectory.Advantages[index];
                newTrajectory.States.RemoveAt(index);
            }
            batches.Add(batch);
        }

        return batches;
    }
}