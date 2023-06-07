using System;
using System.Collections.Generic;
using Physics.Objects.Maths;
using Vector2 = Microsoft.Xna.Framework.Vector2;

namespace Physics.Objects.RigidBodies;

static class ExtensionMethods
{
    public static List<Vector2> ConvertToVector2(this List<Position> positions)
    {
        List<Vector2> vectors = new List<Vector2>();
        foreach (var position in positions)
        {
            vectors.Add(position.GetVector());
        }

        return vectors;
    }
}

public class Skeleton
{
    private List<Position> _positions;
    private Vector2 _centroid;
    private Vector2[] _boundingBox;

    public Skeleton()
    {
        _positions = new List<Position>();
    }

    public void AddPositions(Position[] positions)
    {
        _positions.AddRange(positions);
        _centroid = VectorMath.FindCentroid(_positions.ConvertToVector2().ToArray());
    }

    public Vector2[] GetVectors()
    {
        return _positions.ConvertToVector2().ToArray();
    }

    public Vector2 GetCentroid()
    {
        return _centroid;
    }

    public void Move(Vector2 vector)
    {
        foreach (var position in _positions)
        {
            Vector2 point = position.GetVector();
            position.SetVector(point + vector);
        }
        
        _centroid = VectorMath.FindCentroid(_positions.ConvertToVector2().ToArray());
    }

    public void Rotate(float angle)
    {
        angle = (float) (angle * (Math.PI / 180));
        foreach (var position in _positions)
        {
            Vector2 vector = position.GetVector();
            float x = (float)(Math.Cos(angle) * (vector.X - _centroid.X) - Math.Sin(angle) * (vector.Y - _centroid.Y) + _centroid.X);
            float y = (float) (Math.Sin(angle) * (vector.X - _centroid.X) + Math.Cos(angle) * (vector.Y - _centroid.Y) + _centroid.Y);
            position.SetVector(new Vector2(x, y));
        }
    }
}