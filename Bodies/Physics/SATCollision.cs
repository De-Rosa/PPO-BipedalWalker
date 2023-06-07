using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;

namespace Physics.Bodies.Physics;

public class SATCollision
{
    public static bool IsColliding(List<Vector2> vectorA, List<Vector2> vectorB, Vector2 centroidA, Vector2 centroidB,
        out Vector2 normal, out float depth)
    {
        if (vectorA.Count <= 1 || vectorB.Count <= 1)
            throw new Exception("Cannot calculate collision between shapes with length ≤ 1.");

        normal = Vector2.Zero;
        depth = float.MaxValue;
        
        bool result = AxisChecks(vectorA, vectorB, ref normal, ref depth) &&
                      AxisChecks(vectorB, vectorA, ref normal, ref depth);
        
        // if the shape is pointing towards the object we flip the normal
        Vector2 direction = centroidB - centroidA;
        if (Vector2.Dot(direction, normal) > 0f) normal *= -1;
        
        return result;
    }
    
    private static bool AxisChecks(List<Vector2> vectorA, List<Vector2> vectorB, ref Vector2 normal, ref float depth)
    {
        for (var i = 0; i < vectorA.Count; i++)
        {
            Vector2 edge = vectorA[(i + 1) % vectorA.Count] - vectorA[i];
            Vector2 axis = new Vector2(-edge.Y, edge.X);
            if (axis == Vector2.Zero) continue;
            axis.Normalize();

            Projection projectionA = ProjectPoints(axis, vectorA);
            Projection projectionB = ProjectPoints(axis, vectorB);

            if (!Projection.IsOverlapping(projectionA, projectionB, out float tempDepth)) return false;
            if (tempDepth >= depth) continue;
            
            depth = tempDepth;
            normal = axis;
        }

        return true;
    }

    private static Projection ProjectPoints(Vector2 axis, List<Vector2> vectors)
    {
        var min = float.MaxValue;
        var max = float.MinValue;
        
        foreach (var vector in vectors)
        {
            var projectedVector = Vector2.Dot(axis, vector);
            if (projectedVector < min) min = projectedVector;
            if (projectedVector > max) max = projectedVector;
        }

        return new Projection(min, max);
    }
    
    private struct Projection
    {
        private readonly float _min;
        private readonly float _max;

        public Projection(float min, float max)
        {
            _min = min;
            _max = max;
        }

        public static bool IsOverlapping(Projection projectionA, Projection projectionB, out float depth)
        {
            depth = Math.Min(projectionB._max - projectionA._min, projectionA._max - projectionB._min);
            return (projectionA._min < projectionB._max) && (projectionB._min < projectionA._max);
        }
    }
}   