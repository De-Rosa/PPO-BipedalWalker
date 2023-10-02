using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies.Physics;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Objects.RigidBodies;

// Skeleton class, holds the vectors of a rigid object and it's AABB.
// AABB = Axis Aligned Bounding Box
// The AABB is used for optimisation of physics updates.
public class Skeleton
{
    private List<Vector2> _vectors;
    private Vector2 _centroid;
    private readonly BoundingBox _boundingBox;

    public Skeleton()
    {
        _vectors = new List<Vector2>();
        _boundingBox = new BoundingBox();
    }

    // Checks if two AABBs are colliding (inaccurate but efficient).
    // This is used before SAT collision checking since it is quicker and is always correct
    // at detecting if two objects are NOT colliding.
    public static bool IsColliding(Skeleton skeletonA, Skeleton skeletonB)
    {
        return BoundingBox.IsColliding(skeletonA._boundingBox, skeletonB._boundingBox);
    }

    // Smooths the corners of a given object.
    // Used for creating smooth squares/ovals.
    public void SmoothCorners(int count = 1)
    {
        List<Vector2> newVectors = new List<Vector2>();

        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < _vectors.Count; j++)
            {
                Vector2 faceAB = _vectors[(j + 1) % _vectors.Count] - _vectors[j];
                faceAB *= 0.2f;
                
                Vector2 faceAC = _vectors[ContactPoints.Mod(j - 1, _vectors.Count)] - _vectors[j];
                faceAC *= 0.2f;
                
                newVectors.Add(_vectors[j] + faceAC);
                newVectors.Add(_vectors[j] + faceAB);
            }
            _vectors = newVectors;
            newVectors = new List<Vector2>();
        }
    }

    // Adds a list of vectors to the skeleton.
    public void AddVectors(Vector2[] vectors)
    {
        _vectors.AddRange(vectors);
        _centroid = FindCentroid(_vectors.ToArray());
        _boundingBox.Update(_vectors);
    }

    // Returns the vectors.
    public List<Vector2> GetVectors()
    {
        return _vectors;
    }

    // Returns the centroid of the vectors.
    public Vector2 GetCentroid()
    {
        return _centroid;
    }

    // Moves the skeleton by a given vector.
    public void Move(Vector2 vector)
    {
        for (int i = 0; i < _vectors.Count; i++)
        {
            _vectors[i] += vector;
        }
        
        _centroid += vector;
        _boundingBox.Update(_vectors);
    }

    // Rotates the skeleton by a given angle.
    // The angle is in radians.
    public void Rotate(float angle)
    {
        for (int i = 0; i < _vectors.Count; i++)
        {
            _vectors[i] = Vector2.Transform(_vectors[i] - _centroid, Matrix.CreateRotationZ(angle)) + _centroid;
        }
        
        _boundingBox.Update(_vectors);
    }

    // Finds the centroid from a given set of points.
    private static Vector2 FindCentroid(Vector2[] vectors)
    {
        int count = vectors.Length;
        Vector2 sum = Vector2.Zero;

        foreach (var vector in vectors)
        {
            sum += vector;
        }

        sum /= count;

        return sum;
    }
}

// Bounding box for skeletons, used for collision optimisation.
public class BoundingBox
{
    private Vector2[] _significantCorners;

    public BoundingBox()
    {
        _significantCorners = new Vector2[2];
    }
    
    // Updates the bounding box by finding the significant corners of the shape.
    public void Update(List<Vector2> vectors)
    {
        _significantCorners = FindSignificantCorners(vectors);
    }

    // Finds if two bounding boxes are colliding by checking their significant corners.
    public static bool IsColliding(BoundingBox boundingBoxA, BoundingBox boundingBoxB)
    {
        Vector2[] cornersA = boundingBoxA._significantCorners;
        Vector2[] cornersB = boundingBoxB._significantCorners;

        return (cornersA[0].X < cornersB[1].X && cornersA[1].X > cornersB[0].X &&
                cornersA[0].Y < cornersB[1].Y && cornersA[1].Y > cornersB[0].Y);
    }

    // Finds the significant corners of a shape by finding its maximum and minimum Xs and Ys.
    // An offset can be added/subtracted to make the bounding box bigger/smaller than the actual shape.
    private Vector2[] FindSignificantCorners(List<Vector2> vectors)
    {
        const int offset = 0;
        float maxX = float.MinValue;
        float maxY = float.MinValue;
        float minX = float.MaxValue;
        float minY = float.MaxValue;

        foreach (var position in vectors)
        {
            if (position.X > maxX)
            {
                maxX = position.X;
            }

            if (position.Y > maxY)
            {
                maxY = position.Y;
            }

            if (position.X < minX)
            {
                minX = position.X;
            }

            if (position.Y < minY)
            {
                minY = position.Y;
            }
        }

        return new[] { new Vector2(minX - offset, minY - offset), new Vector2(maxX + offset, maxY + offset) };
    }
}