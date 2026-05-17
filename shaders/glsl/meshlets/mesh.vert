#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;

layout (std140, set = 0, binding = 0) uniform UBOScene
{
	mat4 projection;
	mat4 view;

	vec4 frustumPlanes[6];
	vec4 frozenFrustumPlanes[6];

	vec4 viewPos;
	vec4 lightPos;

	uint features;
	uint padding0;
	uint padding1;
	uint padding2;
} uboScene;

layout (push_constant) uniform PushConsts {
	mat4 model;
} primitive;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outViewVec;
layout (location = 2) out vec3 outLightVec;

void main()
{
	outNormal = inNormal;
	gl_Position = uboScene.projection * uboScene.view * primitive.model * vec4(inPos.xyz, 1.0);

	vec4 pos = uboScene.view * vec4(inPos, 1.0);
	outNormal = mat3(uboScene.view) * inNormal;
	vec3 lPos = mat3(uboScene.view) * uboScene.lightPos.xyz;
	outLightVec = uboScene.lightPos.xyz - pos.xyz;
	outViewVec = uboScene.viewPos.xyz - pos.xyz;
}
