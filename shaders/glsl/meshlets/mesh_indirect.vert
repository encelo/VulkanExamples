#version 460

#extension GL_ARB_shader_draw_parameters : enable

struct Instance
{
	mat4 transform;
	vec4 color;
};

struct Primitive
{
	uint firstMeshlet;
	uint meshletCount;
	uint padding0;
	uint padding1;

	mat4 transform;
	vec3 center;
	float radius;
};

struct Meshlet
{
	uint firstIndex;
	uint indexCount;
	float padding0;
	float padding1;

	vec3 center;
	float radius;

	vec3 coneAxis;
	float coneCutoff;
};

struct VisibleMeshlet
{
	uint instanceIndex;
	uint primitiveIndex;
	uint meshletIndex;
	uint padding0;
};

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;

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
	Meshlet meshlets[ ];
};

layout(std430, set = 1, binding = 3) readonly buffer ssboIn3 {
	VisibleMeshlet visMeshlets[ ];
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
	VisibleMeshlet visMeshlet = visMeshlets[gl_BaseInstance];
	Instance instance = instances[visMeshlet.instanceIndex];
	Primitive primitive = primitives[visMeshlet.primitiveIndex];

	bool meshletColors = (uboScene.features & FEATURE_MESHLET_COLORS) != 0u;
	outColor = meshletColors ? vec4(randomColor(visMeshlet.meshletIndex), 1.0) : instance.color;

	outNormal = inNormal;
	mat4 world = instance.transform * primitive.transform;
	gl_Position = uboScene.projection * uboScene.view * world * vec4(inPos, 1.0);

	vec4 pos = uboScene.view * world * vec4(inPos.xyz, 1.0);;
	outNormal = mat3(uboScene.view) * inNormal;
	vec3 lPos = mat3(uboScene.view) * uboScene.lightPos.xyz;
	outLightVec = uboScene.lightPos.xyz - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}
