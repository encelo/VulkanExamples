/*
* Vulkan Example - glTF scene loading and rendering
*
* Copyright (C) 2020-2026 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

/*
 * Shows how to load and display a simple scene from a glTF file
 * Note that this isn't a complete glTF loader and only basic functions are shown here
 * This means no complex materials, no animations, no skins, etc.
 * For details on how glTF 2.0 works, see the official spec at https://github.com/KhronosGroup/glTF/tree/master/specification/2.0
 *
 * Other samples will load models using a dedicated model loader with more features (see base/VulkanglTFModel.hpp)
 *
 * If you are looking for a complete glTF implementation, check out https://github.com/SaschaWillems/Vulkan-glTF-PBR/
 */

#include <queue>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#ifdef VK_USE_PLATFORM_ANDROID_KHR
	#define TINYGLTF_ANDROID_LOAD_FROM_ASSETS
#endif
#include "tiny_gltf.h"
#include "meshoptimizer.h"

#include "VulkanDebug.h"
#include "vulkanexamplebase.h"

namespace {

// The maximum number of instances in the glTF model
constexpr uint32_t MaxInstanceCount = 128;
// The maximum number of meshlets from all the instances
constexpr uint32_t MaxMeshletCount = MaxInstanceCount * 1024;

enum class DrawMode : int32_t
{
	DRAW_INDEXED,
	DRAW_INDEXED_MESHLETS,
	DRAW_INDIRECT_MESHLETS,
	DRAW_INDIRECT_INSTANCED_MESHLETS
};

int32_t selectedDrawModeGUI = int32_t(DrawMode::DRAW_INDIRECT_INSTANCED_MESHLETS);
const char *drawModeItemsGUI = "Indexed\0Indexed Meshlets\0Indirect Meshlets\0Indirect Instanced Meshlets\0\0";

float meshletsToRenderGUI = 1.0f;
size_t renderedMeshletsCounterGUI = 0;

bool freezeCullingCameraGUI = false;
bool cullPrimitivesGUI = true;
bool cullMeshletsGUI = true;

enum FeatureFlags : uint32_t
{
	CullPrimitives = 1 << 0,
	CullMeshlets = 1 << 1
};

struct Counters
{
	uint32_t meshletCountFromVisibleInstances;
	uint32_t visibleMeshletCount;

	// DispatchIndirectCommand
	uint32_t dispatchX;
	uint32_t dispatchY;
	uint32_t dispatchZ;
};

struct VisibleMeshlet
{
	uint32_t instanceIndex;
	uint32_t primitiveIndex;
	uint32_t meshletIndex;
	uint32_t padding0;
};

struct Sphere
{
	glm::vec3 center;
	float radius;
};

Sphere mergeSpheres(const Sphere &a, const Sphere &b)
{
	glm::vec3 delta = b.center - a.center;
	float distance = glm::length(delta);

	// a contains b
	if (a.radius >= distance + b.radius)
		return a;

	// b contains a
	if (b.radius >= distance + a.radius)
		return b;

	glm::vec3 dir = distance > 0.0001f
		? delta / distance
		: glm::vec3(1, 0, 0);

	glm::vec3 p0 = a.center - dir * a.radius;
	glm::vec3 p1 = b.center + dir * b.radius;

	Sphere result;
	result.center = (p0 + p1) * 0.5f;
	result.radius = glm::length(p1 - result.center);

	return result;
}

}

// Contains everything required to render a glTF model in Vulkan
// This class is heavily simplified (compared to glTF's feature set) but retains the basic glTF structure
class VulkanglTFModel
{
  public:
	// The class requires some Vulkan objects so it can create it's own resources
	vks::VulkanDevice *vulkanDevice;
	VkQueue copyQueue;

	// The vertex layout for the samples' model
	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 normal;
	};

	// Single vertex buffer for all primitives
	struct
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
	} vertices;

	// Single index buffer for all primitives
	struct
	{
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} indices;

	// Single index buffer for all meshlets
	struct
	{
		int count;
		VkBuffer buffer;
		VkDeviceMemory memory;
	} meshletIndices;

	// The following structures roughly represent the glTF scene structure
	// To keep things simple, they only contain those properties that are required for this sample
	struct Node;

	struct Meshlet
	{
		uint32_t firstIndex;
		uint32_t indexCount;
		float padding0;
		float padding1;

		glm::vec3 center;
		float radius;
	};

	// A primitive contains the data for a single draw call
	struct Primitive
	{
		uint32_t firstIndex;
		uint32_t indexCount;
		uint32_t firstMeshlet;
		uint32_t meshletCount;
	};

	// Contains the node's (optional) geometry and can be made up of an arbitrary number of primitives
	struct Mesh
	{
		std::vector<Primitive> primitives;
	};

	// The GPU version of the primitive data
	struct PrimitiveGPU
	{
		uint32_t firstMeshlet;
		uint32_t meshletCount;
		uint32_t padding0;
		uint32_t padding1;

		glm::mat4 transform;

		glm::vec3 center;
		float radius;
	};

	std::vector<PrimitiveGPU> allPrimitivesData;
	std::vector<Meshlet> allMeshletsData;

	// The buffer containing the primitive data
	vks::Buffer allPrimitivesDataBuffer;

	// The buffer containing the meshlet data
	vks::Buffer allMeshletsDataBuffer;

	// The buffer containing the indirect draw commands
	vks::Buffer indirectCommandsBuffer;

	// A node represents an object in the glTF scene graph
	struct Node
	{
		Node *parent;
		std::vector<Node *> children;
		Mesh mesh;
		glm::mat4 matrix;
		~Node()
		{
			for (auto &child : children)
				delete child;
		}
	};

	/*
		Model data
	*/
	std::vector<Node *> nodes;

	~VulkanglTFModel()
	{
		for (auto node : nodes)
		{
			delete node;
		}
		// Release all Vulkan resources allocated for the model
		vkDestroyBuffer(vulkanDevice->logicalDevice, vertices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, vertices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, indices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, indices.memory, nullptr);
		vkDestroyBuffer(vulkanDevice->logicalDevice, meshletIndices.buffer, nullptr);
		vkFreeMemory(vulkanDevice->logicalDevice, meshletIndices.memory, nullptr);

		indirectCommandsBuffer.destroy();
		allPrimitivesDataBuffer.destroy();
		allMeshletsDataBuffer.destroy();
	}

	/*
		glTF loading functions

		The following functions take a glTF input model loaded via tinyglTF and convert all required data into our own structure
	*/

	void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, VulkanglTFModel::Node *parent, std::vector<uint32_t> &indexBuffer, std::vector<VulkanglTFModel::Vertex> &vertexBuffer)
	{
		VulkanglTFModel::Node *node = new VulkanglTFModel::Node{};
		node->matrix = glm::mat4(1.0f);
		node->parent = parent;

		// Get the local node matrix
		// It's either made up from translation, rotation, scale or a 4x4 matrix
		if (inputNode.translation.size() == 3)
			node->matrix = glm::translate(node->matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
		if (inputNode.rotation.size() == 4)
		{
			glm::quat q = glm::make_quat(inputNode.rotation.data());
			node->matrix *= glm::mat4(q);
		}
		if (inputNode.scale.size() == 3)
			node->matrix = glm::scale(node->matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
		if (inputNode.matrix.size() == 16)
			node->matrix = glm::make_mat4x4(inputNode.matrix.data());

		// Load node's children
		if (inputNode.children.size() > 0)
		{
			for (size_t i = 0; i < inputNode.children.size(); i++)
				loadNode(input.nodes[inputNode.children[i]], input, node, indexBuffer, vertexBuffer);
		}

		// If the node contains mesh data, we load vertices and indices from the buffers
		// In glTF this is done via accessors and buffer views
		if (inputNode.mesh > -1)
		{
			const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
			// Iterate through all primitives of this node's mesh
			for (size_t i = 0; i < mesh.primitives.size(); i++)
			{
				const tinygltf::Primitive &glTFPrimitive = mesh.primitives[i];
				uint32_t firstIndex = static_cast<uint32_t>(indexBuffer.size());
				uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
				uint32_t indexCount = 0;
				// Vertices
				{
					const float *positionBuffer = nullptr;
					const float *normalsBuffer = nullptr;
					size_t vertexCount = 0;

					// Get buffer data for vertex positions
					if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
					{
						const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
						const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
						positionBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
						vertexCount = accessor.count;
					}
					// Get buffer data for vertex normals
					if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
					{
						const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
						const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
						normalsBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					}
					// Append data to model's vertex buffer
					for (size_t v = 0; v < vertexCount; v++)
					{
						Vertex vert{
							.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f),
							.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)))
						};
						vertexBuffer.push_back(vert);
					}
				}
				// Indices
				{
					const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.indices];
					const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
					const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];

					indexCount += static_cast<uint32_t>(accessor.count);

					// glTF supports different component types of indices
					switch (accessor.componentType)
					{
						case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
						{
							const uint32_t *buf = reinterpret_cast<const uint32_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
							for (size_t index = 0; index < accessor.count; index++)
								indexBuffer.push_back(buf[index] + vertexStart);
							break;
						}
						case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
						{
							const uint16_t *buf = reinterpret_cast<const uint16_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
							for (size_t index = 0; index < accessor.count; index++)
								indexBuffer.push_back(buf[index] + vertexStart);
							break;
						}
						case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
						{
							const uint8_t *buf = reinterpret_cast<const uint8_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
							for (size_t index = 0; index < accessor.count; index++)
								indexBuffer.push_back(buf[index] + vertexStart);
							break;
						}
						default:
							std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
							return;
					}
				}
				Primitive primitive{
					.firstIndex = firstIndex,
					.indexCount = indexCount,
				};
				node->mesh.primitives.push_back(primitive);
			}
		}

		if (parent)
			parent->children.push_back(node);
		else
			nodes.push_back(node);
	}

	/*
		glTF rendering functions
	*/

	// Draw a single node including child nodes (if present)
	void drawNode(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, VulkanglTFModel::Node *node)
	{
		if (node->mesh.primitives.size() > 0)
		{
			// Pass the node's matrix via push constants
			// Traverse the node hierarchy to the top-most parent to get the final matrix of the current node
			glm::mat4 nodeMatrix = node->matrix;
			VulkanglTFModel::Node *currentParent = node->parent;
			while (currentParent)
			{
				nodeMatrix = currentParent->matrix * nodeMatrix;
				currentParent = currentParent->parent;
			}
			// Pass the final matrix to the vertex shader using push constants
			vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &nodeMatrix);

			for (VulkanglTFModel::Primitive &primitive : node->mesh.primitives)
			{
				vks::debugutils::cmdBeginLabel(commandBuffer, "Primitive");
				if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDEXED))
				{
					if (primitive.indexCount > 0)
						vkCmdDrawIndexed(commandBuffer, primitive.indexCount, 1, primitive.firstIndex, 0, 0);
				}
				else if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDEXED_MESHLETS))
				{
					const size_t numMeshlets = primitive.meshletCount * meshletsToRenderGUI;
					for (size_t meshletIdx = 0; meshletIdx < numMeshlets; meshletIdx++)
					{
						const Meshlet &meshlet = allMeshletsData[primitive.firstMeshlet + meshletIdx];
						vkCmdDrawIndexed(commandBuffer, meshlet.indexCount, 1, meshlet.firstIndex, 0, 0);
					}
					renderedMeshletsCounterGUI += numMeshlets;
				}
				else if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDIRECT_MESHLETS))
				{
					const size_t numMeshlets = primitive.meshletCount * meshletsToRenderGUI;
					vkCmdDrawIndexedIndirect(commandBuffer, indirectCommandsBuffer.buffer,
					                         primitive.firstMeshlet * sizeof(VkDrawIndexedIndirectCommand),
					                         numMeshlets, sizeof(VkDrawIndexedIndirectCommand));
					renderedMeshletsCounterGUI += numMeshlets;
				}
				vks::debugutils::cmdEndLabel(commandBuffer);
			}
		}
		for (auto &child : node->children)
			drawNode(commandBuffer, pipelineLayout, child);
	}

	// Draw the glTF scene starting at the top-level-nodes
	void draw(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout)
	{
		renderedMeshletsCounterGUI = 0;

		// All vertices and indices are stored in single buffers, so we only need to bind once
		VkDeviceSize offsets[1] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertices.buffer, offsets);
		if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDEXED))
			vkCmdBindIndexBuffer(commandBuffer, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
		else
			vkCmdBindIndexBuffer(commandBuffer, meshletIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
		// Render all nodes at top-level
		for (auto &node : nodes)
			drawNode(commandBuffer, pipelineLayout, node);
	}
};

class VulkanExample : public VulkanExampleBase
{
  public:
	bool wireframe = false;

	VulkanglTFModel glTFModel;

	struct Instance
	{
		glm::mat4 transform;
		glm::vec4 color;
	};

	// The array of instances of the glTFModel (CPU-side)
	std::vector<Instance> instances;

	// The buffer containing the instances
	vks::Buffer instancesBuffer;

	// The buffer containing various atomic counters
	vks::Buffer countersBuffer;

	// The buffer containing the meshlets from all instances
	vks::Buffer instanceMeshletsBuffer;

	// The buffer containing the indirect draw commands
	vks::Buffer indirectCommandsBuffer;

	struct UniformData
	{
		glm::mat4 projection;
		glm::mat4 view;

		glm::vec4 frustumPlanes[6];
		glm::vec4 frozenFrustumPlanes[6];

		glm::vec4 viewPos;
		glm::vec4 lightPos = glm::vec4(5.0f, 5.0f, -5.0f, 1.0f);

		uint32_t features;
		uint32_t padding0;
		uint32_t padding1;
		uint32_t padding2;
	} uniformData;
	std::array<vks::Buffer, maxConcurrentFrames> uniformBuffers;

	VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
	struct Pipelines
	{
		VkPipeline solid{ VK_NULL_HANDLE };
		VkPipeline wireframe{ VK_NULL_HANDLE };
	} pipelines;

	struct DescriptorSetLayouts
	{
		VkDescriptorSetLayout matrices{ VK_NULL_HANDLE };
	} descriptorSetLayouts;
	std::array<VkDescriptorSet, maxConcurrentFrames> descriptorSets{};

	struct PipelineHandles
	{
		VkPipelineLayout pipelineLayout{ VK_NULL_HANDLE };
		VkPipeline pipeline{ VK_NULL_HANDLE };
		VkDescriptorSetLayout descriptorSetLayout{ VK_NULL_HANDLE };
		VkDescriptorSet descriptorSet{ VK_NULL_HANDLE };

		void destroy(VkDevice device)
		{
			vkDestroyPipeline(device, pipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		}
	};

	PipelineHandles cullPrimitivesHandles;
	PipelineHandles writeDispatchHandles;
	PipelineHandles cullMeshletsHandles;
	PipelineHandles indirectDrawHandles;

	PipelineHandles indirectCmdsHandles;

	struct CullPrimitivesPush
	{
		uint32_t numInstances;
		uint32_t numPrimitives;
	};

	VkPhysicalDeviceVulkan11Features vulkan11Features{};
	VkPhysicalDeviceVulkan12Features vulkan12Features{};

	VulkanExample() : VulkanExampleBase()
	{
		title = "Indirect meshlets rendering";
		apiVersion = VK_API_VERSION_1_2;

		camera.type = Camera::CameraType::lookat;
		camera.flipY = true;
		camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
		camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);

		// Not checking for support
		vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		vulkan11Features.shaderDrawParameters = VK_TRUE;
		// Not checking for support
		vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		vulkan12Features.drawIndirectCount = VK_TRUE;

		vulkan11Features.pNext = &vulkan12Features;
		deviceCreatepNextChain = &vulkan11Features;
	}

	~VulkanExample()
	{
		if (device)
		{
			vkDestroyPipeline(device, pipelines.solid, nullptr);
			if (pipelines.wireframe != VK_NULL_HANDLE)
				vkDestroyPipeline(device, pipelines.wireframe, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayouts.matrices, nullptr);
			for (auto &buffer : uniformBuffers)
				buffer.destroy();

			cullPrimitivesHandles.destroy(device);
			writeDispatchHandles.destroy(device);
			cullMeshletsHandles.destroy(device);
			indirectDrawHandles.destroy(device);
			indirectCmdsHandles.destroy(device);

			instancesBuffer.destroy();
			countersBuffer.destroy();
			instanceMeshletsBuffer.destroy();
			indirectCommandsBuffer.destroy();
		}
	}

	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid)
			enabledFeatures.fillModeNonSolid = VK_TRUE;

		if (deviceFeatures.multiDrawIndirect)
			enabledFeatures.multiDrawIndirect = VK_TRUE;

		if (deviceFeatures.drawIndirectFirstInstance)
			enabledFeatures.drawIndirectFirstInstance = VK_TRUE;
	}

	void loadglTFFile(std::string filename)
	{
		tinygltf::Model glTFInput;
		tinygltf::TinyGLTF gltfContext;
		std::string error, warning;

		this->device = device;

#if defined(__ANDROID__)
		// On Android all assets are packed with the apk in a compressed form, so we need to open them using the asset manager
		// We let tinygltf handle this, by passing the asset manager of our app
		tinygltf::asset_manager = androidApp->activity->assetManager;
#endif
		bool fileLoaded = gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, filename);

		// Pass some Vulkan resources required for setup and rendering to the glTF model loading class
		glTFModel.vulkanDevice = vulkanDevice;
		glTFModel.copyQueue = queue;

		std::vector<uint32_t> indexBuffer;
		std::vector<VulkanglTFModel::Vertex> vertexBuffer;

		if (fileLoaded)
		{
			const tinygltf::Scene &scene = glTFInput.scenes[0];
			for (size_t i = 0; i < scene.nodes.size(); i++)
			{
				const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
				glTFModel.loadNode(node, glTFInput, nullptr, indexBuffer, vertexBuffer);
			}
		}
		else
		{
			vks::tools::exitFatal("Could not open the glTF file.\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
			return;
		}

		// Create and upload vertex and index buffer
		// We will be using one single vertex buffer and one single index buffer for the whole glTF scene
		// Primitives (of the glTF model) will then index into these using index offsets

		size_t vertexBufferSize = vertexBuffer.size() * sizeof(VulkanglTFModel::Vertex);
		size_t indexBufferSize = indexBuffer.size() * sizeof(uint32_t);
		glTFModel.indices.count = static_cast<uint32_t>(indexBuffer.size());

		// ----- Build meshlets data -----

		std::vector<uint32_t> meshletIndices;
		meshletIndices.reserve(indexBuffer.size());
		size_t meshletIndexBufferSize = 0;
		for (uint32_t nodeIdx = 0; nodeIdx < glTFModel.nodes.size(); nodeIdx++)
		{
			std::queue<VulkanglTFModel::Node *> nodesToVisit;
			nodesToVisit.push(glTFModel.nodes[nodeIdx]);

			while (nodesToVisit.empty() == false)
			{
				VulkanglTFModel::Node *node = nodesToVisit.front();
				nodesToVisit.pop();
				for (uint32_t childIdx = 0; childIdx < node->children.size(); childIdx++)
					nodesToVisit.push(node->children[childIdx]);

				for (uint32_t primitiveIdx = 0; primitiveIdx < node->mesh.primitives.size(); primitiveIdx++)
				{
					VulkanglTFModel::Primitive &primitive = node->mesh.primitives[primitiveIdx];
					const size_t MaxVertices = 64;
					const size_t MaxTriangles = 126;
					const float ConeWeight = 0.0f;

					size_t maxMeshlets = meshopt_buildMeshletsBound(primitive.indexCount, MaxVertices, MaxTriangles);
					std::vector<meshopt_Meshlet> meshlets(maxMeshlets);
					std::vector<unsigned int> meshletVertices(maxMeshlets * MaxVertices);
					std::vector<unsigned char> meshletTriangles(maxMeshlets * MaxTriangles * 3);

					const size_t meshletCount = meshopt_buildMeshlets(meshlets.data(), meshletVertices.data(), meshletTriangles.data(), indexBuffer.data() + primitive.firstIndex,
						primitive.indexCount, &vertexBuffer[0].pos.x, vertexBuffer.size(), sizeof(VulkanglTFModel::Vertex), MaxVertices, MaxTriangles, ConeWeight);

					if (meshletCount == 0)
						continue;
					// Trimming meshlet data vectors
					const meshopt_Meshlet &last = meshlets[meshletCount - 1];
					meshletVertices.resize(last.vertex_offset + last.vertex_count);
					meshletTriangles.resize(last.triangle_offset + last.triangle_count * 3);
					meshlets.resize(meshletCount);

					primitive.firstMeshlet = glTFModel.allMeshletsData.size();
					primitive.meshletCount = meshletCount;

					Sphere primitiveBound;
					for (uint32_t meshletIdx = 0; meshletIdx < meshlets.size(); meshletIdx++)
					{
						const meshopt_Meshlet &m = meshlets[meshletIdx];
						meshopt_optimizeMeshlet(&meshletVertices[m.vertex_offset], &meshletTriangles[m.triangle_offset], m.triangle_count, m.vertex_count);

						const meshopt_Bounds bounds = meshopt_computeMeshletBounds(
							&meshletVertices[m.vertex_offset], &meshletTriangles[m.triangle_offset],
							m.triangle_count, &vertexBuffer[0].pos.x, vertexBuffer.size(), sizeof(VulkanglTFModel::Vertex));

						VulkanglTFModel::Meshlet meshlet;
						meshlet.firstIndex = static_cast<uint32_t>(meshletIndices.size());
						meshlet.indexCount = m.triangle_count * 3;
						meshlet.center = { bounds.center[0], bounds.center[1], bounds.center[2] };
						meshlet.radius = bounds.radius;
						// All meshlets from all primitives of all nodes
						glTFModel.allMeshletsData.push_back(meshlet);

						if (meshletIdx == 0)
						{
							primitiveBound.center = meshlet.center;
							primitiveBound.radius = meshlet.radius;
						}
						else
						{
							const Sphere meshletBound{ meshlet.center, meshlet.radius };
							primitiveBound = mergeSpheres(primitiveBound, meshletBound);
						}

						for (uint32_t triangleIdx = 0; triangleIdx < m.triangle_count; triangleIdx++)
						{
							for (uint32_t vertexIdx = 0; vertexIdx < 3; vertexIdx++)
							{
								const uint32_t localIndex = meshletTriangles[m.triangle_offset + triangleIdx * 3 + vertexIdx];
								const uint32_t absVertexIndex = meshletVertices[m.vertex_offset + localIndex];
								meshletIndices.push_back(absVertexIndex);
							}
						}

						meshletIndexBufferSize += meshlet.indexCount * sizeof(uint32_t);
						glTFModel.meshletIndices.count += static_cast<uint32_t>(meshlet.indexCount);
					}

					VulkanglTFModel::PrimitiveGPU primitiveGpu;
					primitiveGpu.firstMeshlet = primitive.firstMeshlet;
					primitiveGpu.meshletCount = primitive.meshletCount;
					primitiveGpu.transform = node->matrix;
					primitiveGpu.center = primitiveBound.center;
					primitiveGpu.radius = primitiveBound.radius;
					glTFModel.allPrimitivesData.push_back(primitiveGpu);
				}
			}
		}
		assert(glTFModel.allMeshletsData.size() < MaxMeshletCount);

		// ------------------

		vks::Buffer vertexStaging, indexStaging;

		// Create host visible staging buffers (source)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &vertexStaging,
		    vertexBufferSize,
		    vertexBuffer.data()));
		// Index data
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &indexStaging,
		    indexBufferSize,
		    indexBuffer.data()));

		// Create device local buffers (target)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    vertexBufferSize,
		    &glTFModel.vertices.buffer,
		    &glTFModel.vertices.memory));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.vertices.buffer), "Vertex buffer");
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    indexBufferSize,
		    &glTFModel.indices.buffer,
		    &glTFModel.indices.memory));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.indices.buffer), "Index buffer");

		// ----- Meshlet indices -----
		vks::Buffer meshletIndexStaging;

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &meshletIndexStaging,
		    meshletIndexBufferSize,
		    meshletIndices.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    meshletIndexBufferSize,
		    &glTFModel.meshletIndices.buffer,
		    &glTFModel.meshletIndices.memory));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.meshletIndices.buffer), "Meshlet index buffer");

		// ----- Primitive data from the model -----
		size_t allPrimitivesDataBufferSize = glTFModel.allPrimitivesData.size() * sizeof(VulkanglTFModel::PrimitiveGPU);

		vks::Buffer allPrimitivesDataStaging;

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &allPrimitivesDataStaging,
		    allPrimitivesDataBufferSize,
		    glTFModel.allPrimitivesData.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &glTFModel.allPrimitivesDataBuffer,
		    allPrimitivesDataBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.allPrimitivesDataBuffer.buffer), "Primitive data buffer");

		// ----- Meshlet data from all primitives -----
		size_t allMeshletsDataBufferSize = glTFModel.allMeshletsData.size() * sizeof(VulkanglTFModel::Meshlet);

		vks::Buffer allMeshletsDataStaging;

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &allMeshletsDataStaging,
		    allMeshletsDataBufferSize,
		    glTFModel.allMeshletsData.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &glTFModel.allMeshletsDataBuffer,
		    allMeshletsDataBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.allMeshletsDataBuffer.buffer), "Meshlet data buffer");

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    vertexStaging.buffer,
		    glTFModel.vertices.buffer,
		    1,
		    &copyRegion);

		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    indexStaging.buffer,
		    glTFModel.indices.buffer,
		    1,
		    &copyRegion);

		copyRegion.size = meshletIndexBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    meshletIndexStaging.buffer,
		    glTFModel.meshletIndices.buffer,
		    1,
		    &copyRegion);

		copyRegion.size = allPrimitivesDataBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    allPrimitivesDataStaging.buffer,
		    glTFModel.allPrimitivesDataBuffer.buffer,
		    1,
		    &copyRegion);

		copyRegion.size = allMeshletsDataBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    allMeshletsDataStaging.buffer,
		    glTFModel.allMeshletsDataBuffer.buffer,
		    1,
		    &copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		vertexStaging.destroy();
		indexStaging.destroy();

		meshletIndexStaging.destroy();
		allPrimitivesDataStaging.destroy();
		allMeshletsDataStaging.destroy();
	}

	void loadAssets()
	{
		loadglTFFile(getAssetPath() + "models/FlightHelmet/glTF/FlightHelmet.gltf");
	}

	void addInstances()
	{
		instances.reserve(16);

		glm::vec3 position(-1.0f, 0.0f, 0.0f);
		const glm::vec3 scale(1.0f, 1.0f, 1.0f);
		const float angle = glm::radians(180.0f);
		const glm::vec3 axis(0.0f, 1.0f, 0.0f);

		Instance inst;
		inst.transform = glm::mat4(1.0f);
		inst.transform = glm::translate(inst.transform, position);
		inst.transform = glm::rotate(inst.transform, angle, axis);
		inst.transform = glm::scale(inst.transform, scale);
		inst.color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
		instances.push_back(inst);

		position = glm::vec3(0.0f, 0.0f, 0.0f);
		inst.transform = glm::mat4(1.0f);
		inst.transform = glm::translate(inst.transform, position);
		inst.transform = glm::rotate(inst.transform, angle, axis);
		inst.transform = glm::scale(inst.transform, scale);
		inst.color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
		instances.push_back(inst);

		position = glm::vec3(1.0f, 0.0f, 0.0f);
		inst.transform = glm::mat4(1.0f);
		inst.transform = glm::translate(inst.transform, position);
		inst.transform = glm::rotate(inst.transform, angle, axis);
		inst.transform = glm::scale(inst.transform, scale);
		inst.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
		instances.push_back(inst);

		assert(instances.size() <= MaxInstanceCount);
	}

	void setupDescriptorPool()
	{
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxConcurrentFrames),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32)
		};
		// One set for matrices and one for buffers
		const uint32_t maxSetCount = maxConcurrentFrames + 32;
		VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, maxSetCount);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptors()
	{
		/*
			This sample uses separate descriptor sets (and layouts) for the matrices
		*/

		// Descriptor set layout for passing matrices
		VkDescriptorSetLayoutBinding setLayoutBinding = vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0);
		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(&setLayoutBinding, 1);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &descriptorSetLayouts.matrices));

		// Descriptor set for scene matrices per frame, just like the buffers themselves
		for (auto i = 0; i < uniformBuffers.size(); i++)
		{
			VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayouts.matrices, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSets[i]));
			VkWriteDescriptorSet writeDescriptorSet = vks::initializers::writeDescriptorSet(descriptorSets[i], VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers[i].descriptor);
			vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
		}

		// ----- Indirect drawing -----
		{
			// Descriptor set layout for passing storage buffers
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Instance data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
				// Binding 1 : Primitive data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 1),
				// Binding 2 : Meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 2),
				// Binding 3 : Visible meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 3)
			};
			VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &indirectDrawHandles.descriptorSetLayout));

			// Descriptor sets for buffers
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &indirectDrawHandles.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &indirectDrawHandles.descriptorSet));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			// Binding 0 : Instance data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectDrawHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &instancesBuffer.descriptor));
			// Binding 1 : Primitive data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectDrawHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &glTFModel.allPrimitivesDataBuffer.descriptor));
			// Binding 2 : Meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectDrawHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &glTFModel.allMeshletsDataBuffer.descriptor));
			// Binding 3 : Visible meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectDrawHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &instanceMeshletsBuffer.descriptor));
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}
	}

	void preparePipelines()
	{
		// We will use push constants to push the local matrices of a primitive to the vertex shader
		VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(glm::mat4), 0);
		// The pipeline layout uses one descriptor set (set 0 = matrices)
		std::array<VkDescriptorSetLayout, 1> setLayouts = { descriptorSetLayouts.matrices };
		VkPipelineLayoutCreateInfo pipelineLayoutCI{ //} = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
			.pSetLayouts = setLayouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange
		};
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout));

		// Pipeline
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCI = vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);
		VkPipelineRasterizationStateCreateInfo rasterizationStateCI = vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0);
		VkPipelineColorBlendAttachmentState blendAttachmentStateCI = vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);
		VkPipelineColorBlendStateCreateInfo colorBlendStateCI = vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentStateCI);
		VkPipelineDepthStencilStateCreateInfo depthStencilStateCI = vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);
		VkPipelineViewportStateCreateInfo viewportStateCI = vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);
		VkPipelineMultisampleStateCreateInfo multisampleStateCI = vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);
		const std::vector<VkDynamicState> dynamicStateEnables = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
		VkPipelineDynamicStateCreateInfo dynamicStateCI = vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);
		// Vertex input bindings and attributes
		const std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, sizeof(VulkanglTFModel::Vertex), VK_VERTEX_INPUT_RATE_VERTEX),
		};
		const std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, pos)), // Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(VulkanglTFModel::Vertex, normal)) // Location 1: Normal
		};
		VkPipelineVertexInputStateCreateInfo vertexInputStateCI{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size()),
			.pVertexBindingDescriptions = vertexInputBindings.data(),
			.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size()),
			.pVertexAttributeDescriptions = vertexInputAttributes.data(),
		};

		const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
			loadShader(getShadersPath() + "meshlets/mesh.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "meshlets/mesh.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};

		VkGraphicsPipelineCreateInfo pipelineCI{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = static_cast<uint32_t>(shaderStages.size()),
			.pStages = shaderStages.data(),
			.pVertexInputState = &vertexInputStateCI,
			.pInputAssemblyState = &inputAssemblyStateCI,
			.pViewportState = &viewportStateCI,
			.pRasterizationState = &rasterizationStateCI,
			.pMultisampleState = &multisampleStateCI,
			.pDepthStencilState = &depthStencilStateCI,
			.pColorBlendState = &colorBlendStateCI,
			.pDynamicState = &dynamicStateCI,
			.layout = pipelineLayout,
			.renderPass = renderPass,
		};

		// Solid rendering pipeline
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.solid));

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid)
		{
			rasterizationStateCI.polygonMode = VK_POLYGON_MODE_LINE;
			rasterizationStateCI.lineWidth = 1.0f;
			VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &pipelines.wireframe));
		}

		// ----- Indirect drawing pipeline -----

		// The pipeline layout uses two descriptor sets (set 0 = matrices, set 1 = buffers)
		std::array<VkDescriptorSetLayout, 2> setLayoutsInd = { descriptorSetLayouts.matrices, indirectDrawHandles.descriptorSetLayout };
		pipelineLayoutCI = {};
		pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		pipelineLayoutCI.setLayoutCount = static_cast<uint32_t>(setLayoutsInd.size()),
		pipelineLayoutCI.pSetLayouts = setLayoutsInd.data();
		rasterizationStateCI.polygonMode = VK_POLYGON_MODE_FILL;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &indirectDrawHandles.pipelineLayout));

		const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStagesInd = {
			loadShader(getShadersPath() + "meshlets/mesh_indirect.vert.spv", VK_SHADER_STAGE_VERTEX_BIT),
			loadShader(getShadersPath() + "meshlets/mesh_indirect.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT)
		};
		pipelineCI.pStages = shaderStagesInd.data();
		pipelineCI.layout = indirectDrawHandles.pipelineLayout;

		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &indirectDrawHandles.pipeline));
	}

	void setupComputeDescriptors()
	{
		// ----- Cull primitives -----
		{
			// Descriptor set layout for passing storage buffers
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Instance data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
				// Binding 1 : Primitive data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
				// Binding 2 : Meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
				// Binding 3 : Counters
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
				// Binding 4 : Meshlet data from all visible instances
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4)
			};
			VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &cullPrimitivesHandles.descriptorSetLayout));

			// Descriptor sets for buffers
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &cullPrimitivesHandles.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &cullPrimitivesHandles.descriptorSet));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			// Binding 0 : Instance data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullPrimitivesHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &instancesBuffer.descriptor));
			// Binding 1 : Primitive data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullPrimitivesHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &glTFModel.allPrimitivesDataBuffer.descriptor));
			// Binding 2 : Meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullPrimitivesHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &glTFModel.allMeshletsDataBuffer.descriptor));
			// Binding 3 : Counters
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullPrimitivesHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &countersBuffer.descriptor));
			// Binding 4 : Meshlet data from all visible instances
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullPrimitivesHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &instanceMeshletsBuffer.descriptor));
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}

		// ----- Write dispatch counters -----
		{
			// Descriptor set layout for passing storage buffers
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Counters
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0)
			};
			VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &writeDispatchHandles.descriptorSetLayout));

			// Descriptor sets for buffers
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &writeDispatchHandles.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &writeDispatchHandles.descriptorSet));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			// Binding 0 : Counters
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(writeDispatchHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &countersBuffer.descriptor));
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}

		// ----- Cull meshlets -----
		{
			// Descriptor set layout for passing storage buffers
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Instance data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
				// Binding 1 : Primitive data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1),
				// Binding 2 : Meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2),
				// Binding 3 : Visible meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3),
				// Binding 4 : Counters
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4),
				// Binding 5 : Indirect draw commands
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5)
			};
			VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &cullMeshletsHandles.descriptorSetLayout));

			// Descriptor sets for buffers
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &cullMeshletsHandles.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &cullMeshletsHandles.descriptorSet));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			// Binding 0 : Instance data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &instancesBuffer.descriptor));
			// Binding 1 : Primitive data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &glTFModel.allPrimitivesDataBuffer.descriptor));
			// Binding 2 : Meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &glTFModel.allMeshletsDataBuffer.descriptor));
			// Binding 3 : Visible meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3, &instanceMeshletsBuffer.descriptor));
			// Binding 4 : Counters
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4, &countersBuffer.descriptor));
			// Binding 5 : Meshlet data from all visible instances
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(cullMeshletsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5, &indirectCommandsBuffer.descriptor));
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}

		// ----- Indirect commands -----
		{
			// Descriptor set layout for passing storage buffers
			std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
			{
				// Binding 0 : Meshlet data
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0),
				// Binding 1 : Indirect draw commands
				vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1)
			};
			VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI = vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
			VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCI, nullptr, &indirectCmdsHandles.descriptorSetLayout));

			// Descriptor sets for buffers
			const VkDescriptorSetAllocateInfo allocInfo = vks::initializers::descriptorSetAllocateInfo(descriptorPool, &indirectCmdsHandles.descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &indirectCmdsHandles.descriptorSet));

			std::vector<VkWriteDescriptorSet> writeDescriptorSets;
			// Binding 0 : Meshlet data
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectCmdsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &glTFModel.allMeshletsDataBuffer.descriptor));
			// Binding 1 : Indirect draw commands
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(indirectCmdsHandles.descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &glTFModel.indirectCommandsBuffer.descriptor));
			vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, NULL);
		}
	}

	void prepareComputePipelines()
	{
		// ----- Cull primitives -----
		{
			// We will use push constants to push the number of instances
			VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(CullPrimitivesPush), 0);
			// The pipeline layout uses two descriptor sets (set 0 = buffers, set 1 = matrices)
			std::array<VkDescriptorSetLayout, 2> setLayouts = { cullPrimitivesHandles.descriptorSetLayout, descriptorSetLayouts.matrices };
			VkPipelineLayoutCreateInfo pipelineLayoutCI{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pushConstantRange
			};
			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &cullPrimitivesHandles.pipelineLayout));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(cullPrimitivesHandles.pipelineLayout), "Cull primitives - Pipeline layout");

			VkPipelineShaderStageCreateInfo shaderStage = loadShader(getShadersPath() + "meshlets/cull_primitives.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

			VkComputePipelineCreateInfo pipelineCI{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
				.stage = shaderStage,
				.layout = cullPrimitivesHandles.pipelineLayout
			};

			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &cullPrimitivesHandles.pipeline));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, uint64_t(cullPrimitivesHandles.pipeline), "Cull primitives - Pipeline");
		}

		// ----- Write dispatch counters -----
		{
			// The pipeline layout uses two descriptor sets (set 0 = buffers)
			std::array<VkDescriptorSetLayout, 1> setLayouts = { writeDispatchHandles.descriptorSetLayout };
			VkPipelineLayoutCreateInfo pipelineLayoutCI{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
				.pSetLayouts = setLayouts.data(),
			};
			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &writeDispatchHandles.pipelineLayout));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(writeDispatchHandles.pipelineLayout), "Write dispatch - Pipeline layout");

			VkPipelineShaderStageCreateInfo shaderStage = loadShader(getShadersPath() + "meshlets/write_dispatch_counters.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

			VkComputePipelineCreateInfo pipelineCI{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
				.stage = shaderStage,
				.layout = writeDispatchHandles.pipelineLayout
			};

			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &writeDispatchHandles.pipeline));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, uint64_t(writeDispatchHandles.pipeline), "Write dispatch - Pipeline");
		}

		// ----- Cull meshlets -----
		{
			// The pipeline layout uses two descriptor sets (set 0 = buffers, set 1 = matrices)
			std::array<VkDescriptorSetLayout, 2> setLayouts = { cullMeshletsHandles.descriptorSetLayout, descriptorSetLayouts.matrices };
			VkPipelineLayoutCreateInfo pipelineLayoutCI{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
				.pSetLayouts = setLayouts.data()
			};
			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &cullMeshletsHandles.pipelineLayout));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(cullMeshletsHandles.pipelineLayout), "Cull meshlets - Pipeline layout");

			VkPipelineShaderStageCreateInfo shaderStage = loadShader(getShadersPath() + "meshlets/cull_meshlets.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

			VkComputePipelineCreateInfo pipelineCI{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
				.stage = shaderStage,
				.layout = cullMeshletsHandles.pipelineLayout
			};

			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &cullMeshletsHandles.pipeline));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, uint64_t(cullMeshletsHandles.pipeline), "Cull meshlets - Pipeline");
		}

		// ----- Indirect commands -----
		{
			// We will use push constants to push the number of meshlets
			VkPushConstantRange pushConstantRange = vks::initializers::pushConstantRange(VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t), 0);
			// The pipeline layout uses one descriptor set (set 0 = buffers)
			std::array<VkDescriptorSetLayout, 1> setLayouts = { indirectCmdsHandles.descriptorSetLayout };
			VkPipelineLayoutCreateInfo pipelineLayoutCI{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
				.pSetLayouts = setLayouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &pushConstantRange
			};
			VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &indirectCmdsHandles.pipelineLayout));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE_LAYOUT, uint64_t(indirectCmdsHandles.pipelineLayout), "Indirect commands - Pipeline layout");

			VkPipelineShaderStageCreateInfo shaderStage = loadShader(getShadersPath() + "meshlets/indirect.comp.spv", VK_SHADER_STAGE_COMPUTE_BIT);

			VkComputePipelineCreateInfo pipelineCI{
				.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
				.stage = shaderStage,
				.layout = indirectCmdsHandles.pipelineLayout
			};

			VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineCI, nullptr, &indirectCmdsHandles.pipeline));
			vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_PIPELINE, uint64_t(indirectCmdsHandles.pipeline), "Indirect commands - Pipeline");
		}
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		for (auto &buffer : uniformBuffers)
		{
			VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer, sizeof(UniformData), &uniformData));
			VK_CHECK_RESULT(buffer.map());
		}
	}

	void updateUniformBuffers()
	{
		uniformData.projection = camera.matrices.perspective;
		uniformData.view = camera.matrices.view;
		uniformData.viewPos = camera.viewPos;

		const glm::mat4 vp = camera.matrices.perspective * camera.matrices.view;
		glm::vec4 row0(vp[0][0], vp[1][0], vp[2][0], vp[3][0]);
		glm::vec4 row1(vp[0][1], vp[1][1], vp[2][1], vp[3][1]);
		glm::vec4 row2(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
		glm::vec4 row3(vp[0][3], vp[1][3], vp[2][3], vp[3][3]);

		uniformData.frustumPlanes[0] = row3 + row0; // left
		uniformData.frustumPlanes[1] = row3 - row0; // right
		uniformData.frustumPlanes[2] = row3 + row1; // bottom
		uniformData.frustumPlanes[3] = row3 - row1; // top
		uniformData.frustumPlanes[4] = row3 + row2; // near
		uniformData.frustumPlanes[5] = row3 - row2; // far

		for (int i = 0; i < 6; ++i)
		{
			const float len = glm::length(glm::vec3(uniformData.frustumPlanes[i]));
			uniformData.frustumPlanes[i] /= len;
		}

		if (freezeCullingCameraGUI == false)
		{
			for (int i = 0; i < 6; ++i)
				uniformData.frozenFrustumPlanes[i] = uniformData.frustumPlanes[i];
		}

		uniformData.features = 0;
		if (cullPrimitivesGUI)
			uniformData.features |= FeatureFlags::CullPrimitives;
		if (cullMeshletsGUI)
			uniformData.features |= FeatureFlags::CullMeshlets;

		memcpy(uniformBuffers[currentBuffer].mapped, &uniformData, sizeof(UniformData));
	}

	void prepareComputeBuffers()
	{
		// ----- Instances buffer -----
		const size_t instancesBufferSize = MaxInstanceCount * sizeof(Instance);

		vks::Buffer instancesStaging;

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		    &instancesStaging,
		    instancesBufferSize,
		    instances.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &instancesBuffer,
		    instancesBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(instancesBuffer.buffer), "Instances buffer");

		// Copy data from staging buffers (host) do device local buffer (gpu)
		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
		VkBufferCopy copyRegion = {};

		copyRegion.size = instancesBufferSize;
		vkCmdCopyBuffer(
		    copyCmd,
		    instancesStaging.buffer,
		    instancesBuffer.buffer,
		    1,
		    &copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		instancesStaging.destroy();

		// ----- Counters buffer -----
		const size_t countersBufferSize = sizeof(Counters);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    // Needs transfer destination usage so it can be cleared, and indirect usage as it also contains dispatch group counts
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &countersBuffer,
		    countersBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(countersBuffer.buffer), "Counters buffer");

		// ----- Instance meshlets buffer -----
		const size_t instanceMeshletsBufferSize = MaxMeshletCount * sizeof(VisibleMeshlet);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &instanceMeshletsBuffer,
		    instanceMeshletsBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(instanceMeshletsBuffer.buffer), "Instance meshlets buffer");

		// ----- Indirect commands buffer (local to the model, no instances) -----
		const size_t indirectCommandsBufferSizeLocal = glTFModel.allMeshletsData.size() * sizeof(VkDrawIndexedIndirectCommand);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &glTFModel.indirectCommandsBuffer,
		    indirectCommandsBufferSizeLocal,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(glTFModel.indirectCommandsBuffer.buffer), "Indirect commands buffer (local)");

		// ----- Indirect commands buffer -----
		const size_t indirectCommandsBufferSize = MaxMeshletCount * sizeof(VkDrawIndexedIndirectCommand);

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
		    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    &indirectCommandsBuffer,
		    indirectCommandsBufferSize,
		    nullptr));
		vks::debugutils::setObjectName(vulkanDevice->logicalDevice, VK_OBJECT_TYPE_BUFFER, uint64_t(indirectCommandsBuffer.buffer), "Indirect commands buffer");
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		addInstances();
		prepareUniformBuffers();
		prepareComputeBuffers();
		setupDescriptorPool();
		setupDescriptors();
		preparePipelines();
		setupComputeDescriptors();
		prepareComputePipelines();
		prepared = true;
	}

	void buildCommandBuffer()
	{
		VkCommandBuffer cmdBuffer = drawCmdBuffers[currentBuffer];

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2]{};
		clearValues[0].color = {
			{ 0.25f, 0.25f, 0.25f, 1.0f }
		};
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = renderPass,
			.framebuffer = frameBuffers[currentImageIndex],
			.renderArea = { .offset = { .x = 0, .y = 0 }, .extent = { .width = width, .height = height } },
			.clearValueCount = 2,
			.pClearValues = clearValues,
		};

		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));

		if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDIRECT_MESHLETS))
		{
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, indirectCmdsHandles.pipelineLayout, 0, 1, &indirectCmdsHandles.descriptorSet, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, indirectCmdsHandles.pipeline);
			// Pass the total number of meshlets using push constants
			const uint32_t allMeshletsDataCount = static_cast<uint32_t>(glTFModel.allMeshletsData.size());
			vkCmdPushConstants(cmdBuffer, indirectCmdsHandles.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &allMeshletsDataCount);
			const uint32_t groupCountX = (allMeshletsDataCount + 63) / 64;
			vkCmdDispatch(cmdBuffer, groupCountX, 1, 1);

			// ----- BARRIER -----
			VkMemoryBarrier barrier{};
			barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;

			vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
		}
		else if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDIRECT_INSTANCED_MESHLETS))
		{
			// Reset the counters
			vkCmdFillBuffer(cmdBuffer, countersBuffer.buffer, 0, sizeof(Counters), 0);

			// ----- BARRIER -----
			{
				VkBufferMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
				barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

				barrier.buffer = countersBuffer.buffer;
				barrier.offset = 0;
				barrier.size = sizeof(Counters);

				vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
			}
			// ----- -----

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPrimitivesHandles.pipelineLayout, 0, 1, &cullPrimitivesHandles.descriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPrimitivesHandles.pipelineLayout, 1, 1, &descriptorSets[currentBuffer], 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPrimitivesHandles.pipeline);
			// Pass the total number of instances and primitives using push constants
			const CullPrimitivesPush cullPrimitivesPush{ static_cast<uint32_t>(instances.size()), static_cast<uint32_t>(glTFModel.allPrimitivesData.size()) };
			vkCmdPushConstants(cmdBuffer, cullPrimitivesHandles.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPrimitivesPush), &cullPrimitivesPush);
			const uint32_t groupCountX = (cullPrimitivesPush.numInstances * cullPrimitivesPush.numPrimitives + 63) / 64;
			vkCmdDispatch(cmdBuffer, groupCountX, 1, 1);

			// ----- BARRIER -----
			{
				VkMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

				vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
			}
			// ----- -----

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, writeDispatchHandles.pipelineLayout, 0, 1, &writeDispatchHandles.descriptorSet, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, writeDispatchHandles.pipeline);
			vkCmdDispatch(cmdBuffer, 1, 1, 1);

			// ----- BARRIER -----
			{
				VkMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

				vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
			}
			// ----- -----

			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullMeshletsHandles.pipelineLayout, 0, 1, &cullMeshletsHandles.descriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullMeshletsHandles.pipelineLayout, 1, 1, &descriptorSets[currentBuffer], 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullMeshletsHandles.pipeline);
			vkCmdDispatchIndirect(cmdBuffer, countersBuffer.buffer, offsetof(Counters, dispatchX));

			// ----- BARRIER -----
			{
				VkMemoryBarrier barrier{};
				barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
				barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;

				vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
			}
			// ----- -----
		}

		vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		const VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
		vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
		const VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

		if (selectedDrawModeGUI != int32_t(DrawMode::DRAW_INDIRECT_INSTANCED_MESHLETS))
		{
			// Bind scene matrices descriptor to set 0
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframe ? pipelines.wireframe : pipelines.solid);
			glTFModel.draw(cmdBuffer, pipelineLayout);
		}
		else
		{
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectDrawHandles.pipelineLayout, 0, 1, &descriptorSets[currentBuffer], 0, nullptr);
			vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectDrawHandles.pipelineLayout, 1, 1, &indirectDrawHandles.descriptorSet, 0, nullptr);
			vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, indirectDrawHandles.pipeline);
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &glTFModel.vertices.buffer, offsets);
			vkCmdBindIndexBuffer(cmdBuffer, glTFModel.meshletIndices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexedIndirectCount(cmdBuffer, indirectCommandsBuffer.buffer, 0, countersBuffer.buffer, offsetof(Counters, visibleMeshletCount), MaxMeshletCount, sizeof(VkDrawIndexedIndirectCommand));
		}

		drawUI(cmdBuffer);
		vkCmdEndRenderPass(cmdBuffer);
		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
	}

	virtual void render()
	{
		if (!prepared)
			return;
		VulkanExampleBase::prepareFrame();
		updateUniformBuffers();
		buildCommandBuffer();
		VulkanExampleBase::submitFrame();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
			case KEY_F1:
				// Counteract default F1 behavior
				ui.visible = !ui.visible;
				selectedDrawModeGUI = 0;
				break;
			case KEY_F2:
				selectedDrawModeGUI = 1;
				break;
			case KEY_F3:
				selectedDrawModeGUI = 2;
				break;
			case KEY_F4:
				selectedDrawModeGUI = 3;
				break;
			case KEY_F:
				freezeCullingCameraGUI = !freezeCullingCameraGUI;
				break;
		}
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings"))
		{
			if (ImGui::Button("Reset Camera"))
			{
				camera.setPosition(glm::vec3(0.0f, -0.1f, -1.0f));
				camera.setRotation(glm::vec3(0.0f, 45.0f, 0.0f));
				camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
			}

			ImGui::PushItemWidth(200.0f);
			ImGui::Combo("Draw Mode", &selectedDrawModeGUI, drawModeItemsGUI);
			ImGui::PopItemWidth();

			if (selectedDrawModeGUI != int32_t(DrawMode::DRAW_INDIRECT_INSTANCED_MESHLETS))
			{
				if (deviceFeatures.fillModeNonSolid)
					overlay->checkBox("Wireframe", &wireframe);
			}

			if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDEXED_MESHLETS) ||
			    selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDIRECT_MESHLETS))
			{
				if (meshletsToRenderGUI < 0.0f)
					meshletsToRenderGUI = 0.0f;
				else if (meshletsToRenderGUI > 1.0f)
					meshletsToRenderGUI = 1.0f;
				ImGui::SliderFloat("Meshlets to Render", &meshletsToRenderGUI, 0.0f, 1.0f, "%.2f");
				ImGui::Text("Rendered Meshlets: %zu", renderedMeshletsCounterGUI);
			}
			else if (selectedDrawModeGUI == int32_t(DrawMode::DRAW_INDIRECT_INSTANCED_MESHLETS))
			{
				ImGui::Checkbox("Freeze Culling Camera", &freezeCullingCameraGUI);
				ImGui::Checkbox("Cull Primitives", &cullPrimitivesGUI);
				ImGui::Checkbox("Cull Meshlets", &cullMeshletsGUI);
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
