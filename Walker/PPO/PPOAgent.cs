using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Physics.Rendering;
using Physics.Walker.PPO.Network;

namespace Physics.Walker.PPO;

// PPO agent class, acts as the 'brain' of the walker by taking actions and learning based off of them.
// The overall machine learning algorithm used is Proximal Policy Optimisation.
public class PPOAgent
{
    private readonly NeuralNetwork _criticNetwork;
    private readonly NeuralNetwork _actorNetwork;

    private readonly int _stateSize;
    private readonly int _actionSize;
    
    private const string WeightsLocation = "Data/Weights/";

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

    // Parses the neural network structures defined in the hyperparameter settings.
    // Creates the neural networks based on their structure by defining a list of layers.
    private void CreateNetworks(NeuralNetwork critic, NeuralNetwork actor)
    {
        (List<Layer> criticLayers, List<DenseLayer> criticDenseLayers) =
            ParseLayers(Hyperparameters.CriticNeuralNetwork);
        (List<Layer> actorLayers, List<DenseLayer> actorDenseLayers) =
            ParseLayers(Hyperparameters.ActorNeuralNetwork);
        
        critic.AddLayers(criticLayers, criticDenseLayers);
        actor.AddLayers(actorLayers, actorDenseLayers);

        if (criticDenseLayers.Last().GetOutputSize() != 1) throw new Exception("The output size for the final dense layer of the critic is invalid (should be 1).");
        if (actorDenseLayers.Last().GetOutputSize() != _actionSize) throw new Exception($"The output size for the final dense layer of the actor is invalid (should be {_actionSize}).");
    }

    // Uses regex to split the input string into tokens which are then converted into layers.
    private (List<Layer>, List<DenseLayer>) ParseLayers(string structure)
    {
        string pattern = @"^Input( \|\d+\|| \((ReLU|TanH|LeakyReLU)\))+ Output$";
        bool isValid = Regex.IsMatch(structure, pattern, RegexOptions.None);
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
            else
            {
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
                    default:
                        throw new Exception($"Invalid activation layer type: {token}.");
                }
            }
        } 
        return (layers, denseLayers);
    }

    // Saves the current weights of the neural networks to the given file.
    public void Save()
    {
        string[] criticNetwork = _criticNetwork.Save("critic");
        string[] actorNetwork = _actorNetwork.Save("actor");

        File.WriteAllLines($"{Hyperparameters.FilePath}{WeightsLocation}{Hyperparameters.CriticWeightFileName}.txt" , criticNetwork);
        File.WriteAllLines($"{Hyperparameters.FilePath}{WeightsLocation}{Hyperparameters.ActorWeightFileName}.txt" , actorNetwork);

    }
    
    // Trains the neural network.
    // Calculates the returns, value estimates, and advantages, before splitting it into batches and training.
    public void Train(Trajectory trajectory, Renderer renderer)
    {
        CalculateValueEstimates(trajectory);

        if (Hyperparameters.UseGAE)
        {
            GeneralizedAdvantageEstimate(trajectory);
        }
        else
        {
            MonteCarloReturn(trajectory);
            MonteCarloAdvantages(trajectory);
        }
       
        if (Hyperparameters.NormalizeAdvantages) Normalize(trajectory.Advantages);

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
    }

    // PPO training function.
    // Calculates the derivatives of each error and feeds it back into the neural networks.
    // The networks are then optimized using Adam.
    private void Train(Batch batch, out float averageCriticLoss)
    {
        // Zero the networks' gradients
        _actorNetwork.Zero();
        _criticNetwork.Zero();
        
        Matrix std = GetStandardDeviations();
        averageCriticLoss = 0;
        
        // https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
        // Performing gradient accumulation over every timestep in the batch.
        for (int i = 0; i < Hyperparameters.BatchSize; i++)
        {
            // Derivative of the mean squared error: 2(V(s) - G)
            // We re-calculate the value estimate to get cache values.
            float valueEstimate = GetValueEstimate(batch.States[i], true);
            float criticLoss = 2 * (valueEstimate - batch.Returns[i]);
            
            Matrix mean = GetMeanOutput(batch.States[i], true);
            Matrix logProbabilities = GetLogProbabilities(mean, std, batch.Actions[i]);
            
            // Derivative of -LClip with respect to the policy.
            // Equation 20
            Matrix ratio = Matrix.Exponential(logProbabilities - batch.LogProbabilities[i]);
            
            Matrix clippedRatio = Matrix.Clip(ratio, 1f + Hyperparameters.Epsilon, 1f - Hyperparameters.Epsilon);
            Matrix clippedRatioAdvantage = clippedRatio * batch.Advantages[i];
            Matrix ratioAdvantage = ratio * batch.Advantages[i];
            
            Matrix partA = Matrix.LessThan(ratioAdvantage, clippedRatioAdvantage, 1f, 0f);
            partA *= batch.Advantages[i];

            Matrix partB = Matrix.LessThanNotEquals(clippedRatioAdvantage, ratioAdvantage, 1f, 0f);
            partB *= batch.Advantages[i];

            Matrix partC = Matrix.InRange(ratio, 1f + Hyperparameters.Epsilon, 1f - Hyperparameters.Epsilon, 1f, 0f);

            Matrix lClipDerivative = partA + Matrix.HadamardProduct(partB, partC);
            lClipDerivative = Matrix.HadamardDivision(lClipDerivative, Matrix.Exponential(batch.LogProbabilities[i]));
            lClipDerivative *= -1f;

            // Derivative of mu with respect to the policy.
            // Equation 26
            Matrix probabilities = Matrix.Exponential(logProbabilities);
            
            Matrix actionsMinusMean = batch.Actions[i] - mean;
            Matrix variance = Matrix.HadamardProduct(std, std);

            Matrix fraction = Matrix.HadamardDivision(actionsMinusMean, variance);
            Matrix meanDerivative = Matrix.HadamardProduct(probabilities,  fraction);
            
            // Chain rule, dPdMu * dCdP = dCdMu.
            Matrix actorLoss = Matrix.HadamardProduct(meanDerivative, lClipDerivative);

            criticLoss /= Hyperparameters.BatchSize;
            actorLoss /= Hyperparameters.BatchSize;

            averageCriticLoss += criticLoss;
        
            // Feed back the given values to calculate gradients.
            _criticNetwork.FeedBack(Matrix.FromValues(new float[] { criticLoss }));
            _actorNetwork.FeedBack(actorLoss);
        }

        // Optimise the networks based on the gradients received from back propagation.
        _criticNetwork.Optimise();
        _actorNetwork.Optimise();
    }

    // We calculate V(s) which is called the value function. This is the estimate of the return if the AI behaves as expected, and does not take
    // into account the randomness of taking actions.
    private float GetValueEstimate(Matrix state, bool cache = false)
    {
        return _criticNetwork.FeedForward(state, cache).GetValue(0,0);
    }

    // Returns a matrix representation of the float value given in the hyperparameters.
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

    // Samples an action by feeding forward the state and calculating its log probability using the normal PDF. 
    public Matrix SampleActions(Matrix state, out Matrix logProbabilities, out Matrix mean, out Matrix std)
    {
        mean = GetMeanOutput(state);
        std = GetStandardDeviations();
        
        Matrix actions = Matrix.SampleNormal(mean, std);
        logProbabilities = GetLogProbabilities(mean, std, actions);

        return actions;
    }

    // Calculates the PDF of each action in the action matrix given a mean and standard deviation.
    private static Matrix GetLogProbabilities(Matrix mean, Matrix std, Matrix actions)
    {
        return Matrix.LogNormalDensities(mean, std, actions);
    }

    // Returns the output of the actor's neural network.
    private Matrix GetMeanOutput(Matrix state, bool cache = false)
    {
        return _actorNetwork.FeedForward(state, cache);
    }

    // GAE, used for calculating returns and advantages.
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

    // Calculates the delta value required for GAE.
    private float CalculateDelta(Trajectory trajectory, int time, ref float nextValue)
    {
        float currentValue = trajectory.Values[time];
        float delta = trajectory.Rewards[time] + (Hyperparameters.Gamma * nextValue) - currentValue;
        nextValue = currentValue;

        return delta;
    }
    
    // Returns a matrix of value estimates for every state in the trajectory.
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
    // Normalizes a given list. Used for normalizing advantages.
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

    // Basic return (or reward-to-go) calculation.
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

    // Basic advantage calculation using (return - value).
    private void MonteCarloAdvantages(Trajectory trajectory)
    {
        trajectory.Advantages.Clear();
        
        for (int i = 0; i < trajectory.States.Count; i++)
        {
            trajectory.Advantages.Add(trajectory.Returns[i] - trajectory.Values[i]);
        }
    }
    
    // Creates batches for a trajectory by picking random values.
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
}
