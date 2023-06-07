using System;
using System.Collections.Generic;
using System.Reflection.Metadata.Ecma335;
using Microsoft.Xna.Framework;

namespace Physics.Bodies.Physics;

public static class RayCasting
{
    //https://rosettacode.org/wiki/Ray-casting_algorithm
    // TODO: remake this
    public static bool IsColliding(Vector2 point, List<Vector2> vectors, out Vector2 displacedPoint, out Vector2 normal)
    {
        int count = 0;
        displacedPoint = Vector2.Zero;
        float depth = float.MaxValue;
        
        for (int i = 0; i < vectors.Count; i++)
        {
            Vector2 vectorA = vectors[(i + 1) % vectors.Count];
            Vector2 vectorB = vectors[i];
            CheckEdgeAndPoint(point, vectorA, vectorB, ref count, ref depth, ref displacedPoint);
        }

        normal = point - displacedPoint;
        if (normal == Vector2.Zero) return false;
        
        normal.Normalize();
        
        return count % 2 != 0;
    }

    public static bool IsColliding(Vector2 point, List<Vector2> vectors)
    {
        int count = 0;
        for (int i = 0; i < vectors.Count; i++)
        {
            Vector2 vectorA = vectors[(i + 1) % vectors.Count];
            Vector2 vectorB = vectors[i];
            if(RayIntersectsEdge(point, vectorA, vectorB)) count += 1;
        }
        
        return count % 2 != 0;
    }

    private static void CheckEdgeAndPoint(Vector2 point, Vector2 vectorA, Vector2 vectorB, ref int count, ref float depth,
        ref Vector2 displacedPoint)
    {
        if (RayIntersectsEdge(point, vectorA, vectorB)) count += 1;
            
        Vector2 closestPoint = ClosestPointToEdge(point, vectorA, vectorB);
        float distance = Vector2.Distance(point, closestPoint);
        if (distance < depth)
        {
            depth = distance;
            displacedPoint = closestPoint;
        }
    }

    private static bool RayIntersectsEdge(Vector2 point, Vector2 vectorA, Vector2 vectorB)
    {
        const float offset = 0.1f;
        
        if (vectorA.Y > vectorB.Y)
        {
            (vectorB, vectorA) = (vectorA, vectorB);
        }

        if (Math.Abs(point.Y - vectorA.Y) < 0.001f || Math.Abs(point.Y - vectorB.Y) < 0.001f)
        {
            point.Y += offset;
        }

        if (point.Y < vectorA.Y || point.Y > vectorB.Y) return false;
        if (point.X >= Math.Max(vectorA.X, vectorB.X)) return false;
        if (point.X < Math.Min(vectorA.X, vectorB.X)) return true;

        float mRed;
        float mBlue;

        if (Math.Abs(vectorA.X - vectorB.X) > 0.001f)
        {
            mRed = (vectorB.Y - vectorA.Y) / (vectorB.X - vectorA.X);
            mBlue = (point.Y - vectorA.Y) / (point.X - vectorA.X);
        }
        else
        {
            mRed = float.PositiveInfinity;
            mBlue = float.PositiveInfinity;
        }

        return mBlue >= mRed;
    }

    private static Vector2 ClosestPointToEdge(Vector2 point, Vector2 vectorA, Vector2 vectorB)
    {
        Vector2 vectorAP = point - vectorA;
        Vector2 vectorAB = vectorB - vectorA;

        float ABDotAP = Vector2.Dot(vectorAP, vectorAB);
        float distanceToClosestPoint = ABDotAP / vectorAB.LengthSquared();

        if (distanceToClosestPoint < 0) return vectorA;
        if (distanceToClosestPoint > 1) return vectorB;
        return vectorA + (vectorAB * distanceToClosestPoint);
    }
}