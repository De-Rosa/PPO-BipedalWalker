namespace Physics.Materials;

public class SuperRubber : IMaterial
{
    public float Density { get; set; } = 11;
    public float Restitution { get; set; } = 1f;

}