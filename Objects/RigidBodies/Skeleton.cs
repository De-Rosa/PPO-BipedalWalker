using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies.Physics;
using Physics.Objects.Maths;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Objects.RigidBodies;

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

    public static bool IsColliding(Skeleton skeletonA, Skeleton skeletonB)
    {
        return BoundingBox.IsColliding(skeletonA._boundingBox, skeletonB._boundingBox);
    }
    
    public static bool IsColliding(Vector2 point, Skeleton skeleton)
    {
        return BoundingBox.IsColliding(point, skeleton._boundingBox);
    }

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

    public void AddVectors(Vector2[] vectors)
    {
        _vectors.AddRange(vectors);
        _centroid = VectorMath.FindCentroid(_vectors.ToArray());
        _boundingBox.Update(_vectors);
    }

    public List<Vector2> GetVectors()
    {
        return _vectors;
    }

    public Vector2 GetCentroid()
    {
        return _centroid;
    }

    public void Move(Vector2 vector)
    {
        for (int i = 0; i < _vectors.Count; i++)
        {
            _vectors[i] += vector;
        }
        
        _centroid += vector;
        _boundingBox.Update(_vectors);
    }

    public void Rotate(float angle)
    {
        for (int i = 0; i < _vectors.Count; i++)
        {
            _vectors[i] = Vector2.Transform(_vectors[i] - _centroid, Matrix.CreateRotationZ(angle)) + _centroid;
        }
        
        _boundingBox.Update(_vectors);
    }
    
    public void Rotate(float angle, Vector2 point)
    {
        for (int i = 0; i < _vectors.Count; i++)
        {
            _vectors[i] = Vector2.Transform(_vectors[i] - point, Matrix.CreateRotationZ(angle)) + point;
        }
        
        _boundingBox.Update(_vectors);
    }
}

// Bounding box for skeletons, used for collision optimisation
public class BoundingBox
{
    private Vector2[] _significantCorners;

    public BoundingBox()
    {
        _significantCorners = new Vector2[2];
    }
    
    public void Update(List<Vector2> vectors)
    {
        _significantCorners = FindSignificantCorners(vectors);
    }

    public static bool IsColliding(BoundingBox boundingBoxA, BoundingBox boundingBoxB)
    {
        Vector2[] cornersA = boundingBoxA._significantCorners;
        Vector2[] cornersB = boundingBoxB._significantCorners;

        return (cornersA[0].X < cornersB[1].X && cornersA[1].X > cornersB[0].X &&
                cornersA[0].Y < cornersB[1].Y && cornersA[1].Y > cornersB[0].Y);
    }

    public static bool IsColliding(Vector2 point, BoundingBox boundingBox)
    {
        Vector2[] corners = boundingBox._significantCorners;
        return (point.X > corners[0].X && point.X < corners[1].X &&
                point.Y > corners[0].Y && point.Y < corners[1].Y);
    }

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