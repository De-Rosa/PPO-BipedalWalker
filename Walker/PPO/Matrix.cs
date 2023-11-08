using System;
using System.Collections.Generic;

namespace NEA.Walker.PPO;

// Matrix class, implementation of matrix math.
public class Matrix
{
    private readonly int _height; // Rows of matrix
    private readonly int _length; // Columns of matrix
    // We use jagged arrays since they are more efficient than bi-dimensional arrays.
    // https://learn.microsoft.com/en-us/dotnet/fundamentals/code-analysis/quality-rules/ca1814
    private readonly float[][] _values; 

    // Converts a jagged array
    private Matrix(float[][] values)
    {
        _values = values;
        _height = values.Length;
        _length = values[0].Length;
    }
    
    // Creates an empty matrix from a given height and length.
    public static Matrix FromSize(int height, int length)
    {
        List<float[]> list = new List<float[]>();
        for (int i = 0; i < height; i++)
        {
            list.Add(new float[length]);
        }

        return new Matrix(list.ToArray());
    }

    // Converts an array of values into a (X) by 1 matrix.
    public static Matrix FromValues(float[] values)
    {
        Matrix matrix = FromSize(values.Length, 1);
        
        for (int i = 0; i < values.Length; i++)
        {
            matrix.SetValue(i, 0, values[i]);
        }

        return matrix;
    }

    // Xavier, or glorot, initialization is better for training than random uniform values since gradients close to 0 is more efficient.
    // https://stackoverflow.com/questions/53830488/neural-network-only-producing-values-of-1-when-i-add-more-hidden-layers
    // https://datascience.stackexchange.com/questions/102036/where-does-the-normal-glorot-initialization-come-from
    // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    public static Matrix FromXavier(int height, int length)
    {
        Matrix matrix = Matrix.FromSize(height, length);
        Random random = new Random();
        float std = MathF.Sqrt(2f / (height + length));
        
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < length; j++)
            {
                matrix._values[i][j] = NormalDistribution.BoxMullerTransform(0, std, random);
            }
        }

        return matrix;
    }

    // Creates a matrix full of 0s from a given height and length.
    public static Matrix FromZeroes(int height, int length)
    {
        Matrix matrix = Matrix.FromSize(height, length);
        matrix.Zero();
        return matrix;
    }
    
    // Returns the height of the matrix.
    public int GetHeight()
    {
        return _height;
    }

    // Returns the length of the matrix.
    public int GetLength()
    {
        return _length;
    }

    // Loads a matrix from a string of values.
    public static Matrix Load(Matrix matrix, string values)
    {
        float[] floatValues = Array.ConvertAll(values.Split(), float.Parse);
        Matrix newMatrix = Matrix.FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                int index = j + (i * newMatrix._length);
                newMatrix._values[i][j] = floatValues[index];
            }
        }

        return newMatrix;
    }
    
    // Converts the matrix values into a string representation.
    public static string Save(Matrix matrix)
    {
        return string.Join(" ", GetRepresentation(matrix));
    }

    // Converts the matrix values into an array of floats.
    public static float[] GetRepresentation(Matrix matrix)
    {
        float[] floatValues = new float[matrix._length * matrix._height];
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                int index = j + (i * matrix._length);
                floatValues[index] = matrix._values[i][j];
            }
        }

        return floatValues;
    }

    // Sets all of the values inside the matrix to 0.
    public void Zero()
    {
        for (int i = 0; i < _height; i++)
        {
            for (int j = 0; j < _length; j++)
            {
                _values[i][j] = 0;
            }
        }
    }

    // Applies the exponential function to each value in the matrix.
    public static Matrix Exponential(Matrix matrix)
    {
        return PerformOperation(matrix, MathF.Exp);
    }
    
    // Applies the square root function to each value in the matrix.
    public static Matrix SquareRoot(Matrix matrix)
    {
        return PerformOperation(matrix, MathF.Sqrt);
    }
    
    // Matrix multiplication.
    public static Matrix operator * (Matrix matrixA, Matrix matrixB)
    {
        if (matrixA._length != matrixB._height) throw new Exception($"Invalid matrix dimensions: multiplying {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}");
        Matrix newMatrix = FromSize(matrixA._height, matrixB._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = Multiply(i, j, matrixA, matrixB);
            }
        }
        
        return newMatrix;
    }
    
    // Multiplies each value in the matrix with a given float.
    public static Matrix operator * (Matrix matrix, float value)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = matrix._values[i][j] * value;
            }
        }
        
        return newMatrix;
    }
    
    // Divides each value in the matrix by a given float.
    public static Matrix operator / (Matrix matrix, float value)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = matrix._values[i][j] / value;
            }
        }
        
        return newMatrix;
    }
    
    public static Matrix operator * (float value, Matrix matrix)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = matrix._values[i][j] * value;
            }
        }
        
        return newMatrix;
    }
    
    // Adds each value of the two matrices together (pairwise summation).
    public static Matrix operator + (Matrix matrixA, Matrix matrixB)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, adding {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");
        
        Matrix newMatrix = FromSize(matrixA._height, matrixB._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = matrixA._values[i][j] + matrixB._values[i][j];
            }
        }

        return newMatrix;
    }
    
    // Adds a float to each value in the matrix.
    public static Matrix operator + (Matrix matrix, float value)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = matrix._values[i][j] + value;
            }
        }

        return newMatrix;
    }
    
    // Subtracts a float from each value in the matrix.
    public static Matrix operator - (Matrix matrix, float value)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = matrix._values[i][j] - value;
            }
        }

        return newMatrix;
    }
    
    // Subtracts each value of the two matrices together (pairwise subtraction).
    public static Matrix operator - (Matrix matrixA, Matrix matrixB)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, subtracting {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");
        
        Matrix newMatrix = FromSize(matrixA._height, matrixB._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = matrixA._values[i][j] - matrixB._values[i][j];
            }
        }

        return newMatrix;
    }
    
    // Negates all values inside the matrix.
    public static Matrix operator - (Matrix matrix)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[i][j] = -matrix._values[i][j];
            }
        }

        return newMatrix;
    }
    
    // Multiplies each value in the two matrices together (pairwise multiplication).
    public static Matrix HadamardProduct(Matrix matrixA, Matrix matrixB)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, multiplying {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");
        Matrix newMatrix = FromSize(matrixA._height, matrixB._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = matrixA._values[i][j] * matrixB._values[i][j];
            }
        }
        
        return newMatrix;
    }
    
    // Divides each value in the two matrices together (pairwise division).
    public static Matrix HadamardDivision(Matrix matrixA, Matrix matrixB)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, dividing {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");
        Matrix newMatrix = FromSize(matrixA._height, matrixB._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = matrixA._values[i][j] / matrixB._values[i][j];
            }
        }
        
        return newMatrix;
    }

    // Clips the values of the matrix between two values.
    public static Matrix Clip(Matrix matrix, float upper, float lower)
    {
        if (lower > upper) throw new Exception($"Attempting to clip a matrix with an invalid range ({lower} < x < {upper}).");
        
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                if (matrix._values[i][j] >= upper)
                {
                    newMatrix._values[i][j] = upper;
                } else if (matrix._values[i][j] <= lower)
                {
                    newMatrix._values[i][j] = lower;
                }
                else
                {
                    newMatrix._values[i][j] = matrix._values[i][j];
                }
            }
        }

        return newMatrix;
    }
    
    // Sets a given value of the matrix to the 'inRangeValue' if it is inside the given range, 'otherValue' if not.
    public static Matrix InRange(Matrix matrix, float upper, float lower, float inRangeValue, float otherValue)
    {
        if (lower > upper) throw new Exception($"Attempting to compare a matrix with an invalid range ({lower} < x < {upper}).");

        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                if (matrix._values[i][j] >= lower && matrix._values[i][j] <= upper)
                {
                    newMatrix._values[i][j] = inRangeValue;
                }
                else
                {
                    newMatrix._values[i][j] = otherValue;
                }
            }
        }

        return newMatrix;
    }

    // Compares a given value in two matrices, if value A is less than or equal to value B then the value is 'minValue', else 'maxValue'.
    public static Matrix LessThan(Matrix matrixA, Matrix matrixB, float minValue, float maxValue)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, comparing {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");

        Matrix newMatrix = FromSize(matrixA._height, matrixA._length);
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                if (matrixA._values[i][j] <= matrixB._values[i][j])
                {
                    newMatrix._values[i][j] = minValue;
                }
                else
                {
                    newMatrix._values[i][j] = maxValue;
                }
            }
        }

        return newMatrix;
    }
    
    // Compares a given value in two matrices, if value A is less than value B then the value is 'minValue', else 'maxValue'.
    public static Matrix LessThanNotEquals(Matrix matrixA, Matrix matrixB, float minValue, float maxValue)
    {
        if (matrixA._length != matrixB._length || matrixA._height != matrixB._height) throw new Exception($"Invalid matrix dimensions, comparing {matrixA._height}x{matrixA._length} matrix with {matrixB._height}x{matrixB._length}.");

        Matrix newMatrix = FromSize(matrixA._height, matrixA._length);
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                if (matrixA._values[i][j] < matrixB._values[i][j])
                {
                    newMatrix._values[i][j] = minValue;
                }
                else
                {
                    newMatrix._values[i][j] = maxValue;
                }
            }
        }

        return newMatrix;
    }

    // Transposes the matrix, rotating the matrix by 90 degrees (newHeight = length, newLength = height).
    public static Matrix Transpose(Matrix matrix)
    {
        Matrix newMatrix = FromSize(matrix._length, matrix._height);
        for (int i = 0; i < matrix._height; i++)
        {
            for (int j = 0; j < matrix._length; j++)
            {
                newMatrix._values[j][i] = matrix._values[i][j];
            }
        }

        return newMatrix;
    }

    // Flattens a given X by Y matrix to a Y by 1 matrix.
    public static Matrix Flatten(Matrix matrix)
    {
        Matrix newMatrix = FromSize(matrix._height, 1);
        for (int i = 0; i < matrix._height; i++)
        {
            float sum = 0;
            
            for (int j = 0; j < matrix._length; j++)
            {
                sum += matrix._values[i][j];
            }

            newMatrix._values[i][0] = sum;
        }

        return newMatrix;
    }
    
    // Peforms a given function on each value in the matrix.
    public static Matrix PerformOperation (Matrix matrix, Func<float, float> operation)
    {
        Matrix newMatrix = FromSize(matrix._height, matrix._length);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            for (int j = 0; j < newMatrix._length; j++)
            {
                newMatrix._values[i][j] = operation(matrix._values[i][j]);
            }
        }

        return newMatrix;
    }

    public static Matrix SampleNormal(Matrix meanMatrix, Matrix stdMatrix)
    {
        Matrix newMatrix = Matrix.FromSize(meanMatrix._height, 1);
        Random random = new Random();
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            float mean = meanMatrix._values[i][0];
            float std = stdMatrix._values[i][0];

            newMatrix._values[i][0] = NormalDistribution.BoxMullerTransform(mean, std, random);
        }

        return newMatrix;
    }

    public static Matrix LogNormalDensities(Matrix meanMatrix, Matrix stdMatrix, Matrix actionMatrix)
    {
        Matrix newMatrix = Matrix.FromSize(meanMatrix._height, 1);
        
        for (int i = 0; i < newMatrix._height; i++)
        {
            float mean = meanMatrix.GetValue(i, 0);
            float std = stdMatrix.GetValue(i, 0);
            float action = actionMatrix.GetValue(i, 0);

            newMatrix._values[i][0] = NormalDistribution.LogProbabilityDensity(mean, std, action);
        }

        return newMatrix;
    }

    private float[] GetRow(int rowNum)
    {
        return _values[rowNum];
    }
    
    private float[] GetColumn(int columnNum)
    {
        float[] column = new float[_height];
        for (int i = 0; i < _height; i++)
        {
            column[i] = _values[i][columnNum];
        }
        return column;
    }
    
    private static float Multiply(int rowNum, int columnNum, Matrix matrixA, Matrix matrixB)
    {
        float[] rowA = matrixA.GetRow(rowNum);
        float[] columnB = matrixB.GetColumn(columnNum);
        float sum = 0;

        for (int i = 0; i < rowA.Length; i++)
        {
            sum += rowA[i] * columnB[i];
        }

        return sum;
    }

    public float GetValue(int height, int length)
    {
        return _values[height][length];
    }

    public void SetValue(int height, int length, float value)
    {
        _values[height][length] = value;
    }

    public override string ToString()
    {
        return $"{_height}x{_length}";
    }
}
