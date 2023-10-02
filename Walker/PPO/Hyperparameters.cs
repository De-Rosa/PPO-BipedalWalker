using System;
using System.IO;
using System.Text.Json;

namespace Physics.Walker.PPO;

// A non-static version of the hyperparameters class for use in serializing (converting to JSON).
// We need to have a class instance to serialize it, and so this is used when saving/loading JSON files.
[Serializable]
public class SerializableHyperparameters
{
    // Settings
    public bool FastForward { get; set; }
    public bool CollectData { get; set; }
    public bool SaveWeights { get; set; }

    public int Iterations { get; set; }
    public int MaxTimesteps { get; set; }
    public bool RoughFloor { get; set; }

    // Input: 12 inputs, Output: 4 outputs, |X|: a dense layer with an output of X and input of previous in line, (X): an activation function X layer
    public string CriticNeuralNetwork { get; set; }
    public string ActorNeuralNetwork { get; set; }
    
    public string CriticWeightFileName { get; set; }
    public string ActorWeightFileName { get; set; }
    public string FilePath { get; set; }

    // Adam hyperparameters
    // Adam paper states "Good default settings for the tested machine learning problems are alpha=0.001, beta1=0.9, beta2=0.999 and epsilon=1e−8f"
    public float Alpha { get; set; } // learning rate
    public float Beta1 { get; set; } // 1st-order exponential decay
    public float Beta2 { get; set; } // 2nd-order exponential decay
    public float AdamEpsilon { get; set; } // prevent zero division
    
    // Agent hyperparameters
    public int Epochs { get; set; }
    public int BatchSize { get; set; }
    public bool UseGAE { get; set; }
    public bool NormalizeAdvantages { get; set; }
    
    // GAE hyperparameters
    public float Gamma { get; set; } // Discount Factor
    public float Lambda { get; set; } // Smoothing Factor
    
    // PPO hyperparameters
    public float Epsilon { get; set; } // Clipping Factor
    public float LogStandardDeviation { get; set; }

    public SerializableHyperparameters()
    {
        FastForward = Hyperparameters.FastForward;
        CollectData = Hyperparameters.CollectData;
        SaveWeights = Hyperparameters.SaveWeights;
        Iterations = Hyperparameters.Iterations;
        RoughFloor = Hyperparameters.RoughFloor;
        MaxTimesteps = Hyperparameters.MaxTimesteps;
        CriticNeuralNetwork = Hyperparameters.CriticNeuralNetwork;
        ActorNeuralNetwork = Hyperparameters.ActorNeuralNetwork;
        Alpha = Hyperparameters.Alpha;
        Beta1 = Hyperparameters.Beta1;
        Beta2 = Hyperparameters.Beta2;
        AdamEpsilon = Hyperparameters.AdamEpsilon;
        Epochs = Hyperparameters.Epochs;
        BatchSize = Hyperparameters.BatchSize;
        UseGAE = Hyperparameters.UseGAE;
        NormalizeAdvantages = Hyperparameters.NormalizeAdvantages;
        Gamma = Hyperparameters.Gamma;
        Lambda = Hyperparameters.Lambda;
        Epsilon = Hyperparameters.Epsilon;
        LogStandardDeviation = Hyperparameters.LogStandardDeviation;
        CriticWeightFileName = Hyperparameters.CriticWeightFileName;
        ActorWeightFileName = Hyperparameters.ActorWeightFileName;
        FilePath = Hyperparameters.FilePath;
    }
}

// Hyperparameters class, holds all settings and values which can be changed by the user.
public class Hyperparameters
{
    // Settings
    public static bool FastForward = false;
    public static bool CollectData = true;
    public static bool SaveWeights = true;
    public static int Iterations = 100;
    public static int MaxTimesteps = 10000;
    public static bool RoughFloor = false;
    
    // Input: 12 inputs, Output: 4 outputs, |X|: a dense layer with an output of X and input of previous in line, (X): an activation function X layer
    public static string CriticNeuralNetwork = "Input |1| (LeakyReLU) Output";
    public static string ActorNeuralNetwork = "Input |64| (LeakyReLU) |4| (TanH) Output";

    public static string CriticWeightFileName = "critic";
    public static string ActorWeightFileName = "actor";
    public static string FilePath = "/Users/square/Projects/Physics/";

    // Not stored, used specifically for loading weights.
    public static string[] CriticWeights = Array.Empty<string>();
    public static string[] ActorWeights = Array.Empty<string>();

    // Adam hyperparameters
    // Adam paper states "Good default settings for the tested machine learning problems are alpha=0.001, beta1=0.9, beta2=0.999 and epsilon=1e−8f"
    public static float Alpha = 0.001f; // learning rate
    public static float Beta1 = 0.9f; // 1st-order exponential decay
    public static float Beta2 = 0.999f; // 2nd-order exponential decay
    public static float AdamEpsilon = 1e-8f; // prevent zero division
    
    // Agent hyperparameters
    public static int Epochs = 5;
    public static int BatchSize = 64;
    public static bool UseGAE = false;
    public static bool NormalizeAdvantages = true;
    
    // GAE hyperparameters
    public static float Gamma = 0.95f; // Discount Factor
    public static float Lambda = 0.95f; // Smoothing Factor
    
    // PPO hyperparameters
    public static float Epsilon = 0.3f; // Clipping Factor
    public static float LogStandardDeviation = -1f;

    // Converts the hyperparameters into a JSON file.
    public static async void SerializeJson(string fileLocation)
    {
        SerializableHyperparameters hyperparameters = new SerializableHyperparameters();
        
        using FileStream createStream = File.Create(fileLocation);
        await JsonSerializer.SerializeAsync(createStream, hyperparameters, new JsonSerializerOptions { WriteIndented = true });
        await createStream.DisposeAsync();
    }
    
    // Loads the hyperparameters from a JSON file.
    public static async void DeserializeJson(string fileLocation)
    {
        using FileStream openStream = File.OpenRead(fileLocation);
        SerializableHyperparameters hyperparameters = 
            await JsonSerializer.DeserializeAsync<SerializableHyperparameters>(openStream);

        if (hyperparameters != null)
        {
            FastForward = hyperparameters.FastForward;
            CollectData = hyperparameters.CollectData;
            SaveWeights = hyperparameters.SaveWeights;
            Iterations = hyperparameters.Iterations;
            RoughFloor = hyperparameters.RoughFloor;
            MaxTimesteps = hyperparameters.MaxTimesteps;
            CriticNeuralNetwork = hyperparameters.CriticNeuralNetwork;
            ActorNeuralNetwork = hyperparameters.ActorNeuralNetwork;
            FilePath = hyperparameters.FilePath;
            Alpha = hyperparameters.Alpha;
            Beta1 = hyperparameters.Beta1;
            Beta2 = hyperparameters.Beta2;
            AdamEpsilon = hyperparameters.AdamEpsilon;
            Epochs = hyperparameters.Epochs;
            BatchSize = hyperparameters.BatchSize;
            UseGAE = hyperparameters.UseGAE;
            NormalizeAdvantages = hyperparameters.NormalizeAdvantages;
            Gamma = hyperparameters.Gamma;
            Lambda = hyperparameters.Lambda;
            Epsilon = hyperparameters.Epsilon;
            LogStandardDeviation = hyperparameters.LogStandardDeviation;
            CriticWeightFileName = hyperparameters.CriticWeightFileName;
            ActorWeightFileName = hyperparameters.ActorWeightFileName;
        }
    }
    
    // Sets a variable from its given name in a string.
    public static void ReflectionSet(string variableName, object value)
    {
        switch (variableName)
        {
            case "Alpha":
                Alpha = (float) value;
                break;
            case "Beta1":
                Beta1 = (float) value;
                break;
            case "Beta2":
                Beta2 = (float) value;
                break; 
            case "AdamEpsilon":
                AdamEpsilon = (float) value;
                break;
            case "Epochs":
                Epochs = (int) value;
                break;
            case "BatchSize":
                BatchSize = (int) value;
                break;
            case "UseGAE":
                UseGAE = (bool) value;
                break;
            case "NormalizeAdvantages":
                NormalizeAdvantages = (bool) value;
                break;
            case "Gamma":
                Gamma = (float) value;
                break;
            case "Lambda":
                Lambda = (float) value;
                break;
            case "Epsilon":
                Epsilon = (float) value;
                break;
            case "LogStandardDeviation":
                LogStandardDeviation = (float) value;
                break;
            case "FastForward":
                FastForward = (bool) value;
                break;
            case "CollectData":
                CollectData = (bool) value;
                break;
            case "SaveWeights":
                SaveWeights = (bool) value;
                break;
            case "RoughFloor":
                RoughFloor = (bool)value;
                break;
            case "Iterations":
                Iterations = (int) value;
                break;
            case "MaxTimesteps":
                MaxTimesteps = (int) value;
                break;
            case "CriticNeuralNetwork":
                CriticNeuralNetwork = (string) value;
                break;
            case "ActorNeuralNetwork":
                ActorNeuralNetwork = (string) value;
                break;
            case "CriticWeightFileName":
                CriticWeightFileName = (string)value;
                break;
            case "ActorWeightFileName":
                ActorNeuralNetwork = (string) value;
                break;
            case "FilePath":
                FilePath = (string) value;
                break;
            default:
                return;
        }
    }
    
    // Gets the value from a variable's given string.
    public static object ReflectionGet(string variableName)
    {
        switch (variableName)
        {
            case "Alpha":
                return Alpha;
            case "Beta1":
                return Beta1;
            case "Beta2":
                return Beta2;
            case "AdamEpsilon":
                return AdamEpsilon;
            case "Epochs":
                return Epochs;
            case "BatchSize":
                return BatchSize;
            case "UseGAE":
                return UseGAE;
            case "NormalizeAdvantages":
                return NormalizeAdvantages;
            case "Gamma":
                return Gamma;
            case "Lambda":
                return Lambda;
            case "Epsilon":
                return Epsilon;
            case "LogStandardDeviation":
                return LogStandardDeviation;
            case "FastForward":
                return FastForward;
            case "CollectData":
                return CollectData;
            case "SaveWeights":
                return SaveWeights;
            case "RoughFloor":
                return RoughFloor;
            case "Iterations":
                return Iterations;
            case "MaxTimesteps":
                return MaxTimesteps;
            case "ActorNeuralNetwork":
                return ActorNeuralNetwork;
            case "CriticNeuralNetwork":
                return CriticNeuralNetwork;
            case "ActorWeightFileName":
                return ActorWeightFileName;
            case "CriticWeightFileName":
                return CriticWeightFileName;
            case "FilePath":
                return FilePath;
            default:
                return null;
        }
    }
}