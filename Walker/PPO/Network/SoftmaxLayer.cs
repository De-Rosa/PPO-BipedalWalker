
namespace Physics.Walker.PPO.Network;

public class SoftmaxLayer : Layer
{
    public override Matrix FeedForward(Matrix matrix)
    {
        return Softmax(matrix);
    }

    // https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
    // https://stackoverflow.com/questions/54880369/implementation-of-softmax-function-returns-nan-for-high-inputs
    public Matrix Softmax(Matrix matrix)
    {
        Matrix softmax = Matrix.Exponential(matrix); 
        float exponentialSum = softmax.Sum();
        softmax /= exponentialSum;

        return softmax;
    }

    // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
    public override Matrix FeedBack(Matrix matrix, Matrix gradient)
    {
        Matrix softmax = Softmax(gradient);
        Matrix newMatrix = Matrix.FromSize(gradient.GetHeight(), 1);
        
        for (int i = 0; i < matrix.GetHeight(); i++)
        {
            float value = 0;
            if (i == 0)
            {
                value = softmax.GetValue(0, 0);
                value *= 1 - value;
            }
            else
            {
                value = -softmax.GetValue(0, 0);
                value *= softmax.GetValue(i, 0);
            }
            
            newMatrix.SetValue(i, 0, value);
        }

        return newMatrix;
    }

    public override Layer Clone()
    {
        return new SoftmaxLayer();
    }

    public override LayerType GetType()
    {
        return LayerType.SOFTMAX;
    }
}