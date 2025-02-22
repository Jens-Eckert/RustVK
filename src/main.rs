#![allow(
    dead_code,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

// STD includes
use std::collections::HashSet;
use std::ffi::CStr;
use std::fs::File;
use std::mem::size_of;
use std::os::raw::c_void;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

// External includes
use anyhow::{anyhow, Result};
use cgmath::{point3, vec2, vec3, Deg};
use log::*;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use vulkanalia_vma::{self as vma, Alloc};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

// Minimum required macos version that meets the portability requirements.
const MIN_MACOS_VERSION: Version = Version::new(1, 3, 216);

// Debug layers
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 3;

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Atom3D")
        .with_inner_size(LogicalSize::new(800, 600))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };
    let mut mini = false;

    info!("Done!");
    let err = event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() && !mini => {
                    unsafe { app.render(&window) }.unwrap()
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe {
                        app.destroy();
                    }
                }
                WindowEvent::Resized(s) => {
                    if s.width == 0 || s.height == 0 {
                        mini = true;
                    } else {
                        mini = false;
                        app.resized = true;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });

    if let Ok(e) = err {
        debug!("E");
    } else if let Err(e) = err {
        debug!("GREE");
    }

    debug!("Segfault?");

    Ok(())
}

//==========================
// App & AppData
//==========================

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    device: Device,
    data: AppData,
    frame: usize,
    resized: bool,
    start: Instant,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        // Vulkan function loader for current platform
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|e| anyhow!("{}", e))?;
        let mut data = AppData::default();
        let instance = create_vk_instance(window, &entry, &mut data)?;

        data.surface = vk_window::create_surface(&instance, &window, &window)?;

        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        // debug!("Create Swapchain...");
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        // debug!("Create render pass...");
        create_render_pass(&instance, &device, &mut data)?;
        // debug!("Create pipeline...");
        create_descriptor_set_layout(&device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        // debug!("Create framebuffers...");
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        // Create vertex buffer before command buffer because command_buffer depends on vertex buffer.
        create_vertex_buffer(&instance, &device, &mut data)?;
        create_index_buffer(&instance, &device, &mut data)?;
        create_uniform_buffers(&instance, &device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_texture_image(&instance, &device, &mut data)?;
        create_descriptor_sets(&device, &mut data)?;
        create_command_buffers(&device, &mut data)?; // Records command buffers too
        create_sync_objects(&device, &mut data)?;

        Ok(Self {
            entry,
            instance,
            device,
            data,
            frame: 0,
            resized: false,
            start: Instant::now(),
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        // debug!("Frame: {}", self.frame);

        self.device
            .wait_for_fences(&[self.data.in_flight_fences[self.frame]], true, u64::MAX)?;

        let image_index = match self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_sems[self.frame],
            vk::Fence::null(),
        ) {
            Ok((ii, _)) => ii as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        if !self.data.images_in_flight[image_index as usize].is_null() {
            self.device.wait_for_fences(
                &[self.data.images_in_flight[image_index as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index] = self.data.in_flight_fences[self.frame];

        self.update_uniform_buffer(image_index)?;

        let wait_sems = &[self.data.image_available_sems[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_sems = &[self.data.render_finished_sems[self.frame]];
        let sub_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_sems)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_sems);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame]])?;

        // Execute graphics queue command buffers
        self.device.queue_submit(
            self.data.graphics_queue,
            &[sub_info],
            self.data.in_flight_fences[self.frame],
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_sems)
            .swapchains(swapchains)
            .image_indices(image_indices);

        // Present
        let res = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);

        let changed = res == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || res == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = res {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let time = self.start.elapsed().as_secs_f32();

        let model = Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0) * time);
        let view = Mat4::look_at_rh(
            point3(2.0, 2.0, 2.0),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0),
        );
        let mut proj = cgmath::perspective(
            Deg(45.0),
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,
            1000.0,
        );

        // cgmath was made for openGL where the Y coord is flipped, so we have to flip it back.
        proj[1][1] *= -1.0;

        let ubo = UniformBufferObject { model, view, proj };

        let mem = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, mem.cast(), 1);

        self.device
            .unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    /// Destroys our Vulkan app.
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.data.image_available_sems.iter().for_each(|s| {
            self.device.destroy_semaphore(*s, None);
        });

        self.data.render_finished_sems.iter().for_each(|s| {
            self.device.destroy_semaphore(*s, None);
        });

        self.data.in_flight_fences.iter().for_each(|f| {
            self.device.destroy_fence(*f, None);
        });

        self.destroy_swapchain();
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);

        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);

        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);

        self.device
            .destroy_command_pool(self.data.command_pool, None);

        self.device.destroy_device(None);

        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        // Must be last
        self.instance.destroy_instance(None);
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;

        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        create_pipeline(&self.device, &mut self.data)?;
        create_framebuffers(&self.device, &mut self.data)?;
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.data
            .uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));

        self.data
            .uniform_buffers_memory
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));

        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);

        self.data.framebuffers.iter().for_each(|f| {
            self.device.destroy_framebuffer(*f, None);
        });

        self.device
            .free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        self.device.destroy_render_pass(self.data.render_pass, None);
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);

        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    surface: vk::SurfaceKHR,
    messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_sems: Vec<vk::Semaphore>,
    render_finished_sems: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

//==========================
// Instance
//==========================

unsafe fn create_vk_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData,
) -> Result<Instance> {
    let app_info = vk::ApplicationInfo::builder()
        .application_name(b"Atom3D (MVK)\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 4, 304));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but no supported"));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Goofy bullshit for MacOS specific things.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= MIN_MACOS_VERSION {
        info!("Enabling extensions for macos compatability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let mut create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    // Allow debugging of instance creation and destruction
    if VALIDATION_ENABLED {
        create_info = create_info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&create_info, None)?;

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    return Ok(instance);
}

//=========================
// Pipeline
//=========================

unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Describes what will be attached to the framebuffer
    // format: make sure it matches swapchain format decided earlier
    // samples: used for multisampling stuff, not there yet.
    // load_op: What to do with the data in the fb when the render starts, usually clear
    // store_op: What to do after rendering, this is before presentation so make sure it stores it.
    // stencil_load_op: same but with the stencil, which idk what that is yet tbh
    // stencil_store_op: yeah
    // initial_layout: The format of the vk::Image before the pass begins
    // final_layout: The format of the vk::Image after the pass.
    //      - Options are things like PRESENT_SRC_KHR: to be presented in the swapchain, and TRANSFER_DST_OPTIMAL: image will be used as a destination for a memory copy operation
    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // Subpasses are essentially just saying what get passed to the framebuffer.
    // So here, the attachment_refs variable is saying "Hey, you are going to get a SOMETHING with the COLOR_ATTACHMENT format at index 0 of your attachment array"
    // The subpass dependency basically just reaffirms where the subpass will get it's attachments that it's passing.
    let attachment_refs = &[vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(attachment_refs);

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let deps = &[dependency];
    let attachments = &[color_attachment];
    let subpasses = &[subpass];

    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(deps);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    // Loading shaders and creating modules/stages to place into the pipeline.
    let vert = include_bytes!("../shaders/vert.spv");
    let frag = include_bytes!("../shaders/frag.spv");

    let vert_module = create_shader_module(device, vert)?;
    let frag_module = create_shader_module(device, frag)?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(b"main\0");

    // Fixed Pipeline Functions

    //
    let binding_descs = &[Vertex::binding_desc()];
    let attribute_descs = &Vertex::attribute_desc();

    // Describes how vertices are layed out in the buffer and and what attributes are attatched to them.
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descs)
        .vertex_attribute_descriptions(attribute_descs);

    // Describes how to knit vertices together.
    // primitive_restart_enable lets you break up _STRIP type topologies by using the index 0xFFFF or 0xFFFFFFFF.
    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // The depth fields should almost always be 0.0-1.0.
    // They describe the depth of the framebuffer.
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // Scissors are like a crop layer on top of the viewport rect
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let v = &[viewport];
    let s = &[scissor];

    // Combine scissor and viewport into a create info struct
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(v)
        .scissors(s);

    // Rasterizer
    // depth_clamp_enable: if outside depth range, vetices are simply clamped instead of discarded. Useful sometimes.
    // rasterizer_discard_enable: Essentially disables the framebuffer, rendering nothing.
    // polygon_mode: How fragments are generated from geometry. Modes other than FILL require enabling a GPU feature (logical creation)
    // line_width: line thickness in terms of fragments, thciker lines than 1.0 require a GPU feature
    // cull_mode: which faces to cull, determined by front_face
    // front_face: which way vertices go to determine which faces are facing forward
    // depth_bias_enable: adds a constant or bias to the depth value, sometimes used for shadowmapping.
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    // Requires GPU feature enabled in logical device creation.
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

    // How the old framebuffer color and the new fragment colors are blended together.
    // final.rgb = src_alpha * src_color + (1 - src_alpha) * dst_color
    // final.a = src_alpha
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let cb_atts = &[color_blend_attachment];

    // Contains the global color blending operation.
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(cb_atts)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    let set_layouts = &[data.descriptor_set_layout];
    // Layout is used to describe Uniform locations and what they do.
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(set_layouts);

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vert_stage, frag_stage];

    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    // Craeted from a RenderPass object and assigned to each ImageView in the swapchain.
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

//==========================
// Execution Things
//==========================

unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

// Creates and Records command buffers
unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    // Record
    for (i, cb) in data.command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder();

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty())
            .inheritance_info(&inheritance);

        device.begin_command_buffer(*cb, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        device.cmd_begin_render_pass(*cb, &info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(*cb, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
        device.cmd_bind_vertex_buffers(*cb, 0, &[data.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(*cb, data.index_buffer, 0, vk::IndexType::UINT16);
        device.cmd_bind_descriptor_sets(
            *cb,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline_layout,
            0,
            &[data.descriptor_sets[i]],
            &[],
        );
        device.cmd_draw_indexed(*cb, INDICES.len() as u32, 1, 0, 0, 0);
        device.cmd_end_render_pass(*cb);

        device.end_command_buffer(*cb)?;
    }

    Ok(())
}

unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let sem_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_sems
            .push(device.create_semaphore(&sem_info, None)?);
        data.render_finished_sems
            .push(device.create_semaphore(&sem_info, None)?);

        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    // Default to null because no images are being used yet.
    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

//==========================
// Device
//==========================

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= MIN_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder();

    let create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &create_info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);

    Ok(device)
}

use thiserror::Error;

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let props = instance.get_physical_device_properties(physical_device);

        info!("Checking physical device: (`{}`)", props.device_name);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                props.device_name, error
            );
        } else {
            info!("Selected physical device: (`{}`)", props.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device!"))
}

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(&e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions"
        )))
    }
}

//==========================
// Queue Family
//==========================

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let props = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = props
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32); // map here just casts usize to u32

        let present = props
            .iter()
            .enumerate()
            .find(|(i, _)| {
                instance
                    .get_physical_device_surface_support_khr(
                        physical_device,
                        *i as u32,
                        data.surface,
                    )
                    .unwrap()
            })
            .map(|p| p.0 as u32);

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families")))
        }
    }
}

//==========================
// Swapchain
//==========================

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let mut image_count = support.capabilities.min_image_count + 1;

    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&create_info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            let components = vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::IDENTITY)
                .g(vk::ComponentSwizzle::IDENTITY)
                .b(vk::ComponentSwizzle::IDENTITY)
                .a(vk::ComponentSwizzle::IDENTITY);

            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);

            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*i)
                .view_type(vk::ImageViewType::_2D)
                .format(data.swapchain_format)
                .components(components)
                .subresource_range(subresource_range);

            device.create_image_view(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

// Find format with B8G8R8A8_SRGB and SRGB_NONLINEAR, else return first in list
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

// Try to get mailbox mode, default to FIFO
fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

//==========================
// Debug Callback
//==========================

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}): {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}): {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}): {}", type_, message);
    } else {
        trace!("({:?}): {}", type_, message);
    }

    vk::FALSE
}

//==========================
// Vertex Stuff
//==========================

// repr(C) makes sure the structs data is layed out in Memory as it would be in C.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    // How Vertices are given to the Shader
    fn binding_desc() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    // What is inside the Vertex and how is it passed to the shader
    fn attribute_desc() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();

        [pos, color]
    }
}

static VERTICES: [Vertex; 4] = [
    Vertex::new(vec2(-0.5, -0.5), vec3(1.0, 0.0, 0.0)),
    Vertex::new(vec2(0.5, -0.5), vec3(0.0, 1.0, 0.0)),
    Vertex::new(vec2(0.5, 0.5), vec3(0.0, 0.0, 1.0)),
    Vertex::new(vec2(-0.5, 0.5), vec3(1.0, 1.0, 1.0)),
];

static INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

// Generic buffer creation
unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let reqs = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size)
        .memory_type_index(get_memory_type_index(instance, data, props, reqs)?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    dest: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let cb = device.allocate_command_buffers(&info)?[0];

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(cb, &info)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(cb, source, dest, &[regions]);

    device.end_command_buffer(cb)?;

    let cbs = &[cb];
    let info = vk::SubmitInfo::builder().command_buffers(cbs);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;

    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, cbs);

    Ok(())
}

unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    debug!("create_vertex_buffer");

    let size = (size_of::<Vertex>() * VERTICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(VERTICES.as_ptr(), memory.cast(), VERTICES.len());

    device.unmap_memory(staging_buffer_memory);

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        &data,
        size,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let size = (size_of::<u16>() * INDICES.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        &data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(INDICES.as_ptr(), memory.cast(), INDICES.len());

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        &data,
        size,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, &data, staging_buffer, data.index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    props: vk::MemoryPropertyFlags,
    reqs: vk::MemoryRequirements,
) -> Result<u32> {
    let mem = instance.get_physical_device_memory_properties(data.physical_device);

    (0..mem.memory_type_count)
        .find(|i| {
            let suitable = reqs.memory_type_bits & (1 << i) != 0;
            let mem_type = mem.memory_types[*i as usize];
            suitable && mem_type.property_flags.contains(props)
        })
        .ok_or_else(|| anyhow!("failed to find suitable memory type"))
}

//=======================
// Uniform Buffers
//=======================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

unsafe fn create_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0) // Corresponds to layout(binding = 0) in vertex shader
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[ubo_binding];

    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (ub, ubm) = create_buffer(
            instance,
            device,
            &data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(ub);
        data.uniform_buffers_memory.push(ubm);
    }

    Ok(())
}

unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_sizes = &[ubo_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(())
}

//=======================
// Textures
//=======================

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let image = File::open("resources/allears.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;
    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (w, h) = reader.info().size();

    let (staging_buff, staging_mem) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let mem = device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(pixels.as_ptr(), mem.cast(), pixels.len());

    device.unmap_memory(staging_mem);

    Ok(())
}
