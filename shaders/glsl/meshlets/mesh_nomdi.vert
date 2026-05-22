#version 450

struct Instance
{
	mat4 transform;
	vec4 color;
};

struct Primitive
{
	uint firstMeshlet;
	uint meshletCount;
	float padding0;
	float padding1;

	mat4 transform;

	vec3 center;
	float radius;
};

struct IndexLookupData
{
	uint instanceIndex;
	uint primitiveIndex;
	uint meshletIndex;
	uint sourceIndex;
};

struct Vertex
{
	vec3 pos;
	float padding0;
	vec3 normal;
	float padding1;
};

const uint FEATURE_MESHLET_COLORS = 1u << 3;

layout (std140, set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;

	vec4 frustumPlanes[6];
	vec4 frozenFrustumPlanes[6];

	vec4 viewPos;
	vec4 frozenViewPos;

	vec4 lightPos;

	uint features;
	uint padding0;
	uint padding1;
	uint padding2;
} uboScene;

layout(std430, set = 1, binding = 0) readonly buffer ssboIn0 {
	Instance instances[ ];
};

layout(std430, set = 1, binding = 1) readonly buffer ssboIn1 {
	Primitive primitives[ ];
};

layout(std430, set = 1, binding = 2) readonly buffer ssboIn2 {
	IndexLookupData indexLookup[ ];
};

layout(std430, set = 1, binding = 3) readonly buffer ssboIn3
{
	uint indices[];
};

// Used for vertex pulling
layout(std430, set = 1, binding = 4) readonly buffer ssboIn4 {
	Vertex vertices[];
};

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;

vec3 randomColor(uint id)
{
	float h = fract(float(id) * 0.61803398875);
	vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0, 4, 2), 6.0) - 3.0) - 1.0, 0.0, 1.0);
	return mix(vec3(1.0), rgb, 0.7);
}

void main()
{
	IndexLookupData lookup = indexLookup[gl_VertexIndex];
	Instance instance = instances[lookup.instanceIndex];
	Primitive primitive = primitives[lookup.primitiveIndex];

	uint vertexIndex = indices[lookup.sourceIndex];
	Vertex vert = vertices[vertexIndex];

	bool meshletColors = (uboScene.features & FEATURE_MESHLET_COLORS) != 0u;
	outColor = meshletColors ? vec4(randomColor(lookup.meshletIndex), 1.0) : instance.color;

	outNormal = vert.normal;
	mat4 world = instance.transform * primitive.transform;
	gl_Position = uboScene.projection * uboScene.view * world * vec4(vert.pos.xyz, 1.0);

	vec4 pos = uboScene.view * world * vec4(vert.pos, 1.0);
	outNormal = mat3(uboScene.view) * vert.normal;
	vec3 lPos = mat3(uboScene.view) * uboScene.lightPos.xyz;
	outLightVec = uboScene.lightPos.xyz - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}
