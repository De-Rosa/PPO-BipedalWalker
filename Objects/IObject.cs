using System.Collections.Generic;
using Physics.Bodies;

namespace Physics.Objects;

// Object interface, generic class for every object.
public interface IObject
{ 
    public void Update(List<RigidBody> rigidBodies, float deltaTime);
}