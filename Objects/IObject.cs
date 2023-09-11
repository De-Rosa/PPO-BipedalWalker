using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;

namespace Physics.Objects;

public interface IObject
{ 
    public void Update(List<RigidBody> rigidBodies, float deltaTime);
    public IBody GetBody();
}