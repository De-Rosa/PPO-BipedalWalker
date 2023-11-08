using System.Collections.Generic;
using NEA.Bodies;

namespace NEA.Objects;

// Object interface, generic class for every object.
public interface IObject
{ 
    public void Update(List<RigidBody> rigidBodies, float deltaTime);
}