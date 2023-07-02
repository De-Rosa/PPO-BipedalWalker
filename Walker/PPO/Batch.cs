using System;

namespace Physics.Walker.PPO;

public class Batch
{
    public readonly Matrix[] States;
    public readonly Matrix[] LogProbabilities;
    public readonly Matrix[] Actions;
    public readonly Matrix[] Means;
    public readonly Matrix[] Stds;

    public readonly float[] Rewards;
    public readonly float[] Returns;
    public readonly float[] Advantages;
    public readonly float[] Values;

    public int Index;

    public Batch(int batchSize)
    {
        States = new Matrix[batchSize];
        Means = new Matrix[batchSize];
        Stds = new Matrix[batchSize];
        LogProbabilities = new Matrix[batchSize];
        Actions = new Matrix[batchSize];
        Rewards = new float[batchSize];
        Returns = new float[batchSize];
        Values = new float[batchSize];
        Advantages = new float[batchSize];
        Index = 0;
    }
}