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
    private readonly NeuralNetwork _muActorNetwork;
    private readonly NeuralNetwork _sigmaActorNetwork;

    private const int DenseSize = 64;
    private const int Epochs = 5;
    private const int BatchSize = 64;
    private const float Gamma = 0.99f; // Discount Factor
    private const float Lambda = 0.95f; // Smoothing Factor
    private const float Epsilon = 0.2f; // Clipping Factor
    private const float SigmaMax = 10;
    private const float SigmaMin = 0.1f;

    public PPOAgent(int stateSize, int actionSize)
    {
        // Critic neural network transforms a state into a scalar value estimate for use in training the actor
        _criticNetwork = new NeuralNetwork();
        _criticNetwork.AddLayer(new DenseLayer(stateSize, DenseSize));
        _criticNetwork.AddLayer(new LeakyReLULayer());
        _criticNetwork.AddLayer(new DenseLayer(DenseSize, 1));

        // Actor mean neural network transforms a state into an array of means to use in action sampling.
        _muActorNetwork = new NeuralNetwork();
        _muActorNetwork.AddLayer(new DenseLayer(stateSize, DenseSize));
        _muActorNetwork.AddLayer(new LeakyReLULayer());
        _muActorNetwork.AddLayer(new DenseLayer(DenseSize, DenseSize));
        _muActorNetwork.AddLayer(new LeakyReLULayer());
        _muActorNetwork.AddLayer(new DenseLayer(DenseSize, actionSize));
        _muActorNetwork.AddLayer(new TanhLayer());
        
        // Actor sigma neural network transforms a state into an array of standard deviations to use in action sampling.
        _sigmaActorNetwork = new NeuralNetwork();
        _sigmaActorNetwork.AddLayer(new DenseLayer(stateSize, actionSize));
    }

    public void Save(string criticFileLocation, string muFileLocation, string sigmaFileLocation)
    {
        string[] criticNetwork = _criticNetwork.Save();
        string[] muActorNetwork = _muActorNetwork.Save();
        string[] sigmaActorNetwork = _sigmaActorNetwork.Save();

        File.WriteAllLines(criticFileLocation, criticNetwork);
        File.WriteAllLines(muFileLocation, muActorNetwork);
        File.WriteAllLines(sigmaFileLocation, sigmaActorNetwork);

    }

    public void Load(string criticFileLocation, string muFileLocation, string sigmaFileLocation)
    {
        string[] criticNetwork = File.ReadAllLines(criticFileLocation);
        string[] muNetwork = File.ReadAllLines(muFileLocation);
        string[] sigmaNetwork = File.ReadAllLines(muFileLocation);

        if (!File.Exists(criticFileLocation) || !File.Exists(muFileLocation) || !File.Exists(sigmaFileLocation))
        {
            Console.WriteLine("Cannot load weights, the weights files do not exist.");
            return;
        }
        
        _criticNetwork.Load(criticNetwork);
        _muActorNetwork.Load(muNetwork);
        _sigmaActorNetwork.Load(sigmaNetwork);
    }

    public void Render(Renderer renderer)
    {
        _muActorNetwork.Render(renderer);
    }

    public void Train(Trajectory trajectory)
    {
        CalculateValueEstimates(trajectory);
        MonteCarloReturn(trajectory);
        //Standardize(trajectory, trajectory.Returns);
        MonteCarloAdvantages(trajectory);
        
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
        _muActorNetwork.Zero();
        _sigmaActorNetwork.Zero();
        _criticNetwork.Zero();

        valueLoss = 0;
        lClip = 0;
        
        // https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
        // gradient accumulation
        for (int i = 0; i < BatchSize; i++)
        {
            // Derivative of the mean squared error
            // -2(G - V(s))
            float criticLoss = 2 * (GetValueEstimate(batch.States[i]) - batch.Returns[i]);
            criticLoss /= BatchSize;
            valueLoss += criticLoss;
            
            SampleActions(batch.States[i], out Matrix logProbabilities, out Matrix mean, out Matrix std, out Matrix logStd);
            
            // Derivative of L Clip with respect to the policy
            // Equation 20
            Matrix ratio = Matrix.Exponential(logProbabilities - batch.LogProbabilities[i]);
            Matrix clippedRatio = Matrix.Clip(ratio, 1f + Epsilon, 1f - Epsilon);
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];
            Matrix ratioAdvantage = ratio * batch.Advantages[i];

            Matrix lClipMatrix = Matrix.Min(ratioAdvantage, clippedRatioAdvantage);
            float lClipMean = lClipMatrix.Mean();
            lClipMean /= BatchSize;
            lClip += lClipMean;

            Matrix partA = Matrix.Compare(ratioAdvantage, clippedRatioAdvantage, 1f, 0f);
            partA *= batch.Advantages[i];

            Matrix partB = Matrix.CompareNonEquals(clippedRatioAdvantage, ratioAdvantage, 1f, 0f);
            partB *= batch.Advantages[i];

            Matrix partC = Matrix.CompareInRange(ratio, 1f + Epsilon, 1f - Epsilon, 1f, 0f);

            Matrix lClipDerivative = partA + Matrix.HadamardProduct(partB, partC);
            lClipDerivative *= -1f;
            lClipDerivative = Matrix.HadamardDivision(lClipDerivative, Matrix.Exponential(batch.LogProbabilities[i]));

            // Entropy bonus
            lClipDerivative -= Matrix.NormalEntropies(std);
            
            // Derivative of mean with respect to the policy
            // Equation 26
            Matrix probabilities = Matrix.Exponential(logProbabilities);
            
            Matrix actionsMinusMean = batch.Actions[i] - mean;
            Matrix variance = Matrix.HadamardProduct(std, std);

            Matrix meanDerivative = Matrix.HadamardProduct(probabilities,  (Matrix.HadamardDivision(actionsMinusMean, variance)));
            meanDerivative = Matrix.HadamardProduct(meanDerivative, lClipDerivative);
            
            // Derivative of standard deviation with respect to the policy
            // https://docs.google.com/document/d/1FZZvz0JMHKWOOVlXnrmeRMoGpyjqa0m6Q0S2qLECDpA
            Matrix stdCubed = Matrix.HadamardProduct(variance, std);
            Matrix meanSquared = Matrix.HadamardProduct(mean, mean);
            Matrix fraction = batch.Actions[i] - meanSquared - variance;
            fraction = Matrix.HadamardDivision(fraction, stdCubed);

            Matrix transformedDerivative = Matrix.HadamardProduct(fraction, lClipDerivative);
            Matrix stdDerivative = Matrix.HadamardProduct(transformedDerivative, batch.Stds[i]);

            stdDerivative /= BatchSize;
            meanDerivative /= BatchSize;
            
            _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
            _muActorNetwork.FeedBack(meanDerivative);
            _sigmaActorNetwork.FeedBack(stdDerivative);
        }
        
        _criticNetwork.Optimise();
        _muActorNetwork.Optimise();
        _sigmaActorNetwork.Optimise();
    }

    // We calculate V(s) which is called the value function. This is the discounted returns if the AI behaves as expected, and does not take
    // into account the randomness of taking actions
    private float GetValueEstimate(Matrix state)
    {
        return _criticNetwork.FeedForward(state).GetValue(0,0);
    }

    private Matrix GetLogStandardDeviations(Matrix state)
    {
        return _sigmaActorNetwork.FeedForward(state);
    }

    public Matrix SampleActions(Matrix state, out Matrix logProbabilities, out Matrix mean, out Matrix std, out Matrix logStd)
    {
        mean = GetMeanOutput(state);
        logStd = GetLogStandardDeviations(state) - 3.5f;
        std = Matrix.Exponential(logStd);
        std = Matrix.Clip(std, SigmaMax, SigmaMin);
        
        Matrix actions = Matrix.SampleNormal(mean, std);
        logProbabilities = GetLogProbabilities(mean, std, actions);

        return actions;
    }

    private static Matrix GetLogProbabilities(Matrix mean, Matrix std, Matrix actions)
    {
        return Matrix.LogNormalDensities(mean, std, actions);
    }

    private Matrix GetMeanOutput(Matrix state)
    {
        return _muActorNetwork.FeedForward(state);
    }

    // apparently bootstrapping values does not work practically on single workers
    private void GeneralizedAdvantageEstimate(Trajectory trajectory)
    {
        float nextGae = 0;
        float nextValue = 0;
        
        for (int i = trajectory.States.Count - 1; i >= 0; i--)
        {
            float delta = CalculateDelta(trajectory, i, ref nextValue);
            float GAE = delta + (Gamma * Lambda * nextGae);
            trajectory.Advantages.Add(GAE);
            trajectory.Returns.Add(GAE + trajectory.Values[i]);
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