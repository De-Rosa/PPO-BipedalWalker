namespace Physics.Walker.PPO;

public abstract class Layer
{
    public abstract Matrix FeedForward(Matrix matrix);
    public abstract Matrix FeedBack(Matrix matrix, Matrix gradient);
    public abstract Layer Clone();
    public abstract LayerType GetType();
}

public enum LayerType
{
    DENSE,
    ACTIVATION
}