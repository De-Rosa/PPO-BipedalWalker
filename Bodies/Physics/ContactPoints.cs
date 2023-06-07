using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;

namespace Physics.Bodies.Physics;

public class ContactPoints
{
    // Reference: https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics5collisionmanifolds/2017%20Tutorial%205%20-%20Collision%20Manifolds.pdf
    // Uses the algorithm for clipping: https://dyn4j.org/2011/11/contact-points-using-clipping/#cpg-alt
    // TODO: remake this
    public static List<Vector2> GetContactPoints(List<Vector2> vectorA, List<Vector2> vectorB, Vector2 normal)
    {
        Face referenceFace = GetSignificantFace(vectorA, normal);
        Vector2 rfVector = referenceFace.VectorB - referenceFace.VectorA;
        
        Face incidentFace = GetSignificantFace(vectorB, -normal);
        Vector2 ifVector = incidentFace.VectorB - incidentFace.VectorA;

        if (Math.Abs(Vector2.Dot(rfVector, normal)) > Math.Abs(Vector2.Dot(ifVector, normal)))
        {
            (referenceFace, incidentFace) = (incidentFace, referenceFace);
            rfVector = referenceFace.VectorB - referenceFace.VectorA;
        }

        rfVector.Normalize();

        float offset = Vector2.Dot(rfVector, referenceFace.VectorA);

        List<Vector2> clippedPoints = ClipVectors(incidentFace.VectorA, incidentFace.VectorB, rfVector, offset);
        if (clippedPoints.Count < 2) return new List<Vector2> { };

        offset = Vector2.Dot(rfVector, referenceFace.VectorB);
        clippedPoints = ClipVectors(clippedPoints[0], clippedPoints[1], -rfVector, -offset);
        if (clippedPoints.Count < 2) return new List<Vector2> { };

        Vector2 referenceNormal = new Vector2(rfVector.Y, -rfVector.X);

        float maximum = Vector2.Dot(referenceNormal, referenceFace.Max);
    
        if (Vector2.Dot(referenceNormal, clippedPoints.First()) - maximum < 0)
        {
            clippedPoints.Remove(clippedPoints.First());
        }

        if (Vector2.Dot(referenceNormal, clippedPoints.Last()) - maximum < 0)
        {
            clippedPoints.Remove(clippedPoints.Last());
        }

        return clippedPoints;
    }

    private static List<Vector2> ClipVectors(Vector2 vectorA, Vector2 vectorB, Vector2 normal, float offset)
    {
        List<Vector2> points = new List<Vector2>();
        float distanceA = Vector2.Dot(vectorA, normal) - offset;
        float distanceB = Vector2.Dot(vectorB, normal) - offset;
        
        if (distanceA >= 0) points.Add(vectorA);
        if (distanceB >= 0) points.Add(vectorB);
        
        if (distanceA * distanceB < 0)
        {
            Vector2 edge = vectorB - vectorA;
            float location = distanceA / (distanceA - distanceB);
            edge *= location;
            edge += vectorA;
            
            points.Add(edge);
        }

        return points;
    }

    private static Face GetSignificantFace(List<Vector2> vectors, Vector2 normal)
    {
        Vector2 significantVertex = GetSignificantVertex(vectors, normal, out int index);
        
        Vector2 afterFace = significantVertex - vectors[(index + 1) % vectors.Count];
        afterFace.Normalize();
        Vector2 beforeFace = significantVertex - vectors[Mod((index - 1), vectors.Count)];
        beforeFace.Normalize();
        
        if (Vector2.Dot(normal, beforeFace) >= Vector2.Dot(normal, afterFace))
        {
            return new Face(significantVertex, vectors[Mod((index - 1), vectors.Count)], significantVertex);
        }

        return new Face(vectors[(index + 1) % vectors.Count], significantVertex, significantVertex);
    }

    private static Vector2 GetSignificantVertex(List<Vector2> vectors, Vector2 normal, out int index)
    {
        Vector2 significantVertex = Vector2.Zero;
        index = -1;
        float minimumDistance = float.MaxValue;

        for (int i = 0; i < vectors.Count; i++)
        {
            float projection = Vector2.Dot(vectors[i], normal);
            if (!(projection < minimumDistance)) continue;
            significantVertex = vectors[i];
            index = i;
            minimumDistance = projection;
        }
        
        return significantVertex;
    }

    private struct Face
    {
        public readonly Vector2 VectorA;
        public readonly Vector2 VectorB;
        public readonly Vector2 Max;

        public Face(Vector2 vectorA, Vector2 vectorB, Vector2 max)
        {
            VectorA = vectorA;
            VectorB = vectorB;
            Max = max;
        }
    }
    
    public static int Mod(float a, float b)
    {
        return Convert.ToInt32(a - b * Math.Floor(a / b));
    }
}