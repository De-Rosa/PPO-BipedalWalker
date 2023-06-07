using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects.RigidBodies;

namespace Physics.Objects.SoftBodies;

public class Square : SoftBody, IObject
{
    private Square(List<Point> points, List<Spring> springs) : base(points, springs) {}

    public static Square FromSize(IMaterial material, Vector2 start, int count, float spacing)
    {
        Point[][] points = new Point[count][];
        List<Spring> springs = new List<Spring>();

        for (int j = 0; j < count; j++)
        {
            points[j] = new Point[count];

            for (int i = 0; i < count; i++)
            {
                Vector2 position = new Vector2(start.X + (i * spacing), start.Y + (j * spacing));
                Point point = new Point(material, position, spacing * 0.2f);
                points[j][i] = point;
            }
        }

        List<Point> newPoints = new List<Point>();
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < count; j++)
            {
                newPoints.Add(points[j][i]);
            }
        }

        for (int x = 0; x < count; x++)
        {
            for (int y = 0; y < count; y++)
            {
                if (x < count - 1)
                {
                    springs.Add(new Spring(points[y][x], points[y][x+1]));
                    if (y < count - 1)
                    {
                        springs.Add(new Spring(points[y][x], points[y+1][x+1]));
                    }
                }

                if (y >= count - 1) continue;
                springs.Add(new Spring(points[y][x], points[y+1][x]));
                if (x > 0)
                {
                    springs.Add(new Spring(points[y][x], points[y+1][x-1]));
                }
            }
        }

        return new Square(newPoints, springs);
    }

    public IBody GetBody()
    {
        return this;
    }

    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime)
    {
        Step(rigidBodies, softBodies, deltaTime);
    }
}