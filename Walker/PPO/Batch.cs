namespace Physics.Walker.PPO;

public class Batch
{
    public readonly Matrix[] States;
    public readonly float[] Rewards;
    public readonly float[] Returns;
    public readonly float[] Advantages;
    public readonly Matrix[] Probabilities;

    public Batch(int batchSize)
    {
        States = new Matrix[batchSize];
        Rewards = new float[batchSize];
        Returns = new float[batchSize];
        Advantages = new float[batchSize];
        Probabilities = new Matrix[batchSize];
    }
}