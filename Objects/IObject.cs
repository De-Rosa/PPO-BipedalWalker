using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;

namespace Physics.Objects.RigidBodies;

public interface IObject
{ 
    public void Update(List<IObject> objects, float deltaTime);
    public List<Vector2> GetVectors();
    public RigidBody GetBody();
}