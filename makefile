shaders: shaders/vertex.vert shaders/fragment.frag
	/Users/jenseckert/VulkanSDK/1.4.304.1/macOS/bin/glslc shaders/vertex.vert -o shaders/vert.spv
	/Users/jenseckert/VulkanSDK/1.4.304.1/macOS/bin/glslc shaders/fragment.frag -o shaders/frag.spv

run:
	cargo run