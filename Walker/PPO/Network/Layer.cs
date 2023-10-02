namespace Physics.Walker.PPO;

// Abstract layer class, parent of each layer.
// Used as a generic class for each type.
public abstract class Layer
{
    public abstract Matrix FeedForward(Matrix matrix);
    public abstract Matrix FeedBack(Matrix matrix, Matrix gradient);
}
