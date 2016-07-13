using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.IO;
using System.Runtime.InteropServices;
using System.Drawing;

using Vulkan;                     // Core Vulkan classes
using Vulkan.Managed;             // A managed interface to Vulkan
using Vulkan.Managed.ObjectModel; // Extentions to object handles

using Image     = Vulkan.Image;
using Buffer    = Vulkan.Buffer;
using Semaphore = Vulkan.Semaphore;

using OpenTK;
using OpenTK.Graphics;
using OpenTK.Input;

namespace OpenTK.VulkanExample
{
    public class RotatingCubeExample : GameWindow
    {
        #region Classes

        public class VertexData
        {
            public Buffer Buffer;
            public DeviceMemory DeviceMemory;
            public VertexInputBindingDescription[] BindingDescriptions;
            public VertexInputAttributeDescription[] AttributeDescriptions;
            public int[] Indicies;
            public Buffer IndexBuffer;
            public DeviceMemory IndexDeviceMemory;
        }

        public class ImageData
        {
            public uint Width;
            public uint Height;
            public Image Image;
            public DeviceMemory Memory;
            public ImageView View;
            public Sampler Sampler;
        }

        public class SwapchainData
        {
            public SwapchainKHR Swapchain;
            public List<ImageData> Images;
            public List<Framebuffer> Framebuffers;
            public Format ImageFormat;
        }

        class UniformData
        {
            public Buffer Buffer;
            public DeviceMemory Memory;
            public DescriptorBufferInfo Descriptor;
            public uint AllocSize;
        }

        struct Transform
        {
            public Matrix4 projection;
            public Matrix4 model;
            public Matrix4 view;
        }

        #endregion

        Device device;
        
        PhysicalDeviceMemoryProperties physDeviceMem;
        DebugReportCallbackEXT debugCallback;

        ISurfaceProvider surfaceProvider;

        // --

        Instance instance;
        PhysicalDevice[] physDevices;
        PhysicalDevice physDevice;
        QueueFamilyProperties[] queueFamilies;
        //Device device;
        Queue queue;
        CommandPool cmdPool;

        SurfaceKHR surface;
        
        ImageData textureData;
        UniformData uniformData;
        Transform transform;
        VertexData vertexData;
        ImageData depthStencil;
        List<PipelineShaderStageCreateInfo> shaderInfos;
        SwapchainData swapchainData;
        Image[] swapchainImages;
        RenderPass renderPass;
        DescriptorPool descriptorPool;
        DescriptorSetLayout descriptorSetLayout;
        DescriptorSet descriptorSet;
        PipelineLayout pipelineLayout;
        Pipeline[] pipelines;
        Pipeline pipeline;

        CommandBuffer[] cmdBuffers;

        double cubeRotation;
        double cubeRotationSpeed = 50;
        Vector3 camPosition = new Vector3(2, 0.75f, -2);
        float camYRotation = 45;

        float movSpd = 1;
        float rotSpd = 10;

        #region Contructors

        public RotatingCubeExample()
            : this(640, 480, GraphicsMode.Default, "OpenTK Game Window", 0, DisplayDevice.Default)
        {
        }
        
        public RotatingCubeExample(int width, int height)
            : this(width, height, GraphicsMode.Default, "OpenTK Game Window", 0, DisplayDevice.Default)
        {
        }
        
        public RotatingCubeExample(int width, int height, GraphicsMode mode)
            : this(width, height, mode, "OpenTK Game Window", 0, DisplayDevice.Default)
        {
        }
        
        public RotatingCubeExample(int width, int height, GraphicsMode mode, string title)
            : this(width, height, mode, title, 0, DisplayDevice.Default)
        {
        }
        
        public RotatingCubeExample(int width, int height, GraphicsMode mode, string title, GameWindowFlags options)
            : this(width, height, mode, title, options, DisplayDevice.Default)
        {
        }
        
        public RotatingCubeExample(int width, int height, GraphicsMode mode, string title, GameWindowFlags options, DisplayDevice device)
            : this(width, height, mode, title, options, device, 1, 0)
        {
        }
        
        public RotatingCubeExample(int width, int height, GraphicsMode mode, string title, GameWindowFlags options, DisplayDevice device, int major, int minor)
            : base(width, height, mode, title, options, device)
        {
        }

        #endregion

        #region GameWindow

        protected override void OnLoad(EventArgs e)
        {
            // Surface provider
            var impl  = GetImplementation();
            var win32 = (OpenTK.Platform.Windows.WinGLNative)impl;
            surfaceProvider = new Win32SurfaceProvider(win32);
            // ---

            String[] instanceEnabledLayers = new string[]
            {
                "VK_LAYER_LUNARG_standard_validation"
            };

            var instanceEnabledExtensions = new[]
            {
                VulkanConstant.KhrSurfaceExtensionName,
                VulkanConstant.KhrWin32SurfaceExtensionName,
                VulkanConstant.ExtDebugReportExtensionName,
            };

            instance      = CreateInstance(instanceEnabledLayers, instanceEnabledExtensions);
            debugCallback = DebugUtils.CreateDebugReportCallback(instance, DebugReport);
            physDevices   = EnumeratePhysicalDevices(instance);
            physDevice    = physDevices.First();
            queueFamilies = physDevice.GetQueueFamilyProperties();
            physDeviceMem = physDevice.GetMemoryProperties();

            surface = surfaceProvider.CreateSurface(instance);
            var supportsPresent = physDevice.GetWin32PresentationSupportKHR(0);
            var supportsSurface = physDevice.GetSurfaceSupportKHR(0, surface);

            if(!(supportsPresent && supportsSurface))
                throw new InvalidDataException();

            String[] deviceEnabledLayers = new string[]
            {
                "VK_LAYER_LUNARG_standard_validation"
            };

            var deviceEnabledExtensions = new[]
            {
                VulkanConstant.KhrSwapchainExtensionName,
            };

            device = CreateDevice(physDevice, 0, deviceEnabledLayers, deviceEnabledExtensions);
            queue = GetQueue(physDevice, 0);
            cmdPool = CreateCommandPool(0);

            textureData = LoadTexture("./test-image.png", queue, cmdPool);

            uniformData = CreateUniformBuffer(typeof(Transform));
            transform = UpdateTransform(new Transform(), Width, Height, -3);
            CopyTransform(transform, uniformData);
            
            vertexData = CreateVertexData();

            var stencilFormat = GetDepthStencilFormat(physDevice);
            depthStencil = CreateDepthStencil(stencilFormat, (uint)Width, (uint)Height);
            SetStencilLayout(cmdPool, queue, depthStencil);
            
            shaderInfos = new List<PipelineShaderStageCreateInfo>();
            shaderInfos.Add(GetShaderStageCreateInfo(ShaderStageFlags.Vertex, "./Shaders/texture.vert.spv"));
            shaderInfos.Add(GetShaderStageCreateInfo(ShaderStageFlags.Fragment, "./Shaders/texture.frag.spv"));
            
            swapchainData = CreateSwapchain(physDevice, surface, (uint)Width, (uint)Height);
            swapchainImages = device.GetSwapchainImagesKHR(swapchainData.Swapchain);
            swapchainData.Images = InitializeSwapchainImages(queue, cmdPool, swapchainImages, swapchainData.ImageFormat, (uint)Width, (uint)Height);

            renderPass = CreateRenderPass(swapchainData.ImageFormat, stencilFormat);

            swapchainData.Framebuffers = swapchainData.Images
                .Select(x => CreateFramebuffer(renderPass, x, depthStencil))
                .ToList();

            descriptorPool = CreateDescriptorPool();
            descriptorSetLayout = CreateDescriptorSetLayout();
            descriptorSet = CreateDescriptorSet(descriptorPool, descriptorSetLayout, textureData, uniformData);

            pipelineLayout = CreatePipelineLayout(descriptorSetLayout);
            pipelines = CreatePipelines(pipelineLayout, renderPass, shaderInfos.ToArray(), vertexData);
            pipeline = pipelines.First();

            cmdBuffers = AllocateCommandBuffers(cmdPool, (uint)swapchainData.Images.Count());
            for(int x = 0; x < swapchainData.Images.Count(); x++)
                CreateCommandBuffer(cmdBuffers[x], swapchainData.Images[x], swapchainData.Framebuffers[x]);

            base.OnLoad(e);
        }
        
        protected override void OnUpdateFrame(FrameEventArgs e)
        {
            if(Keyboard[Key.W])
                camPosition.Z += movSpd * (float)e.Time;

            if(Keyboard[Key.S])
                camPosition.Z += -movSpd * (float)e.Time;

            if(Keyboard[Key.A])
                camPosition.X += movSpd * (float)e.Time;

            if(Keyboard[Key.D])
                camPosition.X += -movSpd * (float)e.Time;

            if(Keyboard[Key.Q])
                camPosition.Y += movSpd * (float)e.Time;

            if(Keyboard[Key.E])
                camPosition.Y += -movSpd * (float)e.Time;

            if(Keyboard[Key.Z])
                camYRotation -= rotSpd * (float)e.Time;

            if(Keyboard[Key.X])
                camYRotation += rotSpd * (float)e.Time;

            if(Keyboard[Key.F])
                Console.WriteLine(camPosition);

            if(Keyboard[Key.R])
                camPosition = new Vector3(0, 0, 0);
            
            if(Keyboard[Key.Escape])
            {
                Exit();
            }

            var pos = Matrix4.CreateTranslation(camPosition);
            pos *= Matrix4.CreateRotationY(DegreesToRadians(camYRotation));
            
            transform.view = pos;
            //transform.view *= Matrix4.CreateRotationX(DegreesToRadians(90));
            CopyTransform(transform, uniformData);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            var semaphoreCreateInfo = new SemaphoreCreateInfo();
            var presentSemaphore = device.CreateSemaphore(semaphoreCreateInfo);
            semaphoreCreateInfo.Dispose();

            var currentBufferIndex = (int)device.AcquireNextImageKHR(swapchainData.Swapchain, ulong.MaxValue, presentSemaphore, null);
            SubmitForExecution(queue, presentSemaphore, cmdBuffers[currentBufferIndex]);

            var presentInfo = new PresentInfoKHR(new[]{ swapchainData.Swapchain }, new[]{ (uint)currentBufferIndex });
            queue.PresentKHR(presentInfo);
            presentInfo.Dispose();

            queue.WaitIdle();

            device.DestroySemaphore(presentSemaphore);
        }

        protected override void OnClosed(EventArgs e)
        {
            device.DestroyBuffer(uniformData.Buffer);
            device.FreeMemory(uniformData.Memory);

            device.FreeDescriptorSets(descriptorPool, descriptorSet);
            device.DestroyDescriptorSetLayout(descriptorSetLayout);
            device.DestroyDescriptorPool(descriptorPool);

            device.DestroyImageView(depthStencil.View);
            device.DestroyImage(depthStencil.Image);
            device.FreeMemory(depthStencil.Memory);

            device.DestroySampler(textureData.Sampler);
            device.DestroyImageView(textureData.View);
            device.DestroyImage(textureData.Image);
            device.FreeMemory(textureData.Memory);

            device.FreeCommandBuffers(cmdPool, cmdBuffers);

            device.DestroyShaderModule(shaderInfos[0].Module);
            device.DestroyShaderModule(shaderInfos[1].Module);

            swapchainData.Images.ForEach(x => device.DestroyImageView(x.View));
            swapchainData.Framebuffers.ForEach(x => device.DestroyFramebuffer(x));
            device.DestroySwapchainKHR(swapchainData.Swapchain);

            device.DestroyPipeline(pipeline);
            device.DestroyPipelineLayout(pipelineLayout);
            device.DestroyRenderPass(renderPass);

            device.DestroyBuffer(vertexData.IndexBuffer);
            device.FreeMemory(vertexData.IndexDeviceMemory);
            device.DestroyBuffer(vertexData.Buffer);
            device.FreeMemory(vertexData.DeviceMemory);

            device.DestroyCommandPool(cmdPool);

            device.Destroy();

            instance.DestroySurfaceKHR(surface);

            DebugUtils.DestroyDebugReportCallback(instance, debugCallback);

            instance.Destroy();

            base.OnClosed(e);
        }

        #endregion

        #region Rendering

        void CreateCommandBuffer(CommandBuffer cmdBuffer, ImageData swapchainImageData, Framebuffer framebuffer)
        {
            var beginInfo = new CommandBufferBeginInfo();
            cmdBuffer.Begin(beginInfo);
            beginInfo.Dispose();

            PipelineBarrierSetLayout(cmdBuffer, swapchainImageData.Image, ImageLayout.Undefined, ImageLayout.ColorAttachmentOptimal, AccessFlags.None, AccessFlags.ColorAttachmentWrite);

            var clearRange = new ImageSubresourceRange(ImageAspectFlags.Color, 0, 1, 0, 1);
            cmdBuffer.ClearColorImage(swapchainImageData.Image, ImageLayout.TransferDstOptimal, new ClearColorValue(), clearRange);

            var stencilRange = new ImageSubresourceRange(ImageAspectFlags.Depth | ImageAspectFlags.Stencil, 0, 1, 0, 1);
            cmdBuffer.ClearDepthStencilImage(depthStencil.Image, ImageLayout.TransferDstOptimal, new ClearDepthStencilValue(1.0f, 0), stencilRange);

            RenderTexturedQuad(cmdBuffer, vertexData, swapchainImageData, pipelineLayout, descriptorSet, renderPass, pipeline, framebuffer, swapchainImageData.Width, swapchainImageData.Height);

            PipelineBarrierSetLayout(cmdBuffer, swapchainImageData.Image, ImageLayout.ColorAttachmentOptimal, ImageLayout.PresentSrcKHR, AccessFlags.ColorAttachmentWrite, AccessFlags.MemoryRead);

            cmdBuffer.End();
        }

        void RenderTexturedQuad(CommandBuffer cmdBuffer, VertexData vertexData, ImageData imageData, PipelineLayout pipelineLayout, DescriptorSet descriptorSet, RenderPass renderPass, Pipeline pipeline, Framebuffer framebuffer, uint width, uint height)
        {
            var viewport = new Viewport(0, 0, width, height, 0, 1);
            cmdBuffer.SetViewport(0, viewport);

            var renderArea = new Rect2D(new Offset2D(0, 0), new Extent2D(width, height));
            //var clearValues = new[] { new ClearValue { Color = new ClearColorValue() }, new ClearValue { DepthStencil = new ClearDepthStencilValue(1.0f, 0) } };
            var renderPassBegin = new RenderPassBeginInfo(renderPass, framebuffer, renderArea, null);
            //renderPassBegin.ClearValues = clearValues;
            cmdBuffer.BeginRenderPass(renderPassBegin, SubpassContents.Inline);
            renderPassBegin.Dispose();

            cmdBuffer.BindDescriptorSets(PipelineBindPoint.Graphics, pipelineLayout, 0, new[] { descriptorSet }, null);

            cmdBuffer.BindPipeline(PipelineBindPoint.Graphics, pipeline);

            cmdBuffer.BindVertexBuffers(0, new[] { vertexData.Buffer }, new DeviceSize[] { 0 });
            cmdBuffer.BindIndexBuffer(vertexData.IndexBuffer, 0, IndexType.Uint32);
            cmdBuffer.DrawIndexed((uint)vertexData.Indicies.Length, 1, 0, 0, 1);

            cmdBuffer.EndRenderPass();
        }

        #endregion

        #region Vulkan

        #region Assets

        VertexData CreateVertexData()
        {
            var data = new VertexData();

            var quadVertices = new[,]
            {
                {  0.5f, -0.5f,  0.5f,  0f, 0f,   0f,  0f,  1f },
                { -0.5f, -0.5f,  0.5f,  1f, 0f,   0f,  0f,  1f },
                {  0.5f,  0.5f,  0.5f,  0f, 1f,   0f,  0f,  1f },
                { -0.5f,  0.5f,  0.5f,  1f, 1f,   0f,  0f,  1f },
                {  0.5f,  0.5f, -0.5f,  0f, 1f,   0f,  1f,  0f },
                { -0.5f,  0.5f, -0.5f,  1f, 1f,   0f,  1f,  0f },
                {  0.5f, -0.5f, -0.5f,  0f, 1f,   0f,  0f, -1f },
                { -0.5f, -0.5f, -0.5f,  1f, 1f,   0f,  0f, -1f },
                {  0.5f,  0.5f,  0.5f,  0f, 0f,   0f,  1f,  0f },
                { -0.5f,  0.5f,  0.5f,  1f, 0f,   0f,  1f,  0f },
                {  0.5f,  0.5f, -0.5f,  0f, 0f,   0f,  0f, -1f },
                { -0.5f,  0.5f, -0.5f,  1f, 0f,   0f,  0f, -1f },
                {  0.5f, -0.5f, -0.5f,  0f, 0f,   0f, -1f,  0f },
                {  0.5f, -0.5f,  0.5f,  0f, 1f,   0f, -1f,  0f },
                { -0.5f, -0.5f,  0.5f,  1f, 1f,   0f, -1f,  0f },
                { -0.5f, -0.5f, -0.5f,  1f, 0f,   0f, -1f,  0f },
                { -0.5f, -0.5f,  0.5f,  0f, 0f,  -1f,  0f,  0f },
                { -0.5f,  0.5f,  0.5f,  0f, 1f,  -1f,  0f,  0f },
                { -0.5f,  0.5f, -0.5f,  1f, 1f,  -1f,  0f,  0f },
                { -0.5f, -0.5f, -0.5f,  1f, 0f,  -1f,  0f,  0f },
                {  0.5f, -0.5f, -0.5f,  0f, 0f,   1f,  0f,  0f },
                {  0.5f,  0.5f, -0.5f,  0f, 1f,   1f,  0f,  0f },
                {  0.5f,  0.5f,  0.5f,  1f, 1f,   1f,  0f,  0f },
                {  0.5f, -0.5f,  0.5f,  1f, 0f,   1f,  0f,  0f },
            };

            DeviceSize memorySize = (ulong)(sizeof(float) * quadVertices.Length);
            data.Buffer = CreateBuffer(memorySize, BufferUsageFlags.VertexBuffer);

            var memoryRequirements = device.GetBufferMemoryRequirements(data.Buffer);
            var memoryIndex = FindMemoryIndex(MemoryPropertyFlags.HostVisible);
            var allocateInfo = new MemoryAllocateInfo(memoryRequirements.Size, memoryIndex);
            data.DeviceMemory = BindBuffer(data.Buffer, allocateInfo);

            var vertexPtr = device.MapMemory(data.DeviceMemory, 0, memorySize);
            VulkanUtils.Copy2DArray(quadVertices, vertexPtr, memorySize, memorySize);
            device.UnmapMemory(data.DeviceMemory);

            data.Indicies = new[]
            {
                 0, 2, 3, // right
                 0, 3, 1,
                 8, 4, 5, // bottom
                 8, 5, 9,
                10, 6, 7, // left
                10, 7,11,
                12,13,14, // top
                12,14,15,
                16,17,18, // back
                16,18,19,
                20,21,22, // front
                20,22,23,
            };

            memorySize = (ulong)(sizeof(uint) * data.Indicies.Length);
            data.IndexBuffer = CreateBuffer(memorySize, BufferUsageFlags.IndexBuffer);

            memoryRequirements = device.GetBufferMemoryRequirements(data.IndexBuffer);
            memoryIndex = FindMemoryIndex(MemoryPropertyFlags.HostVisible);
            allocateInfo = new MemoryAllocateInfo(memoryRequirements.Size, memoryIndex);
            data.IndexDeviceMemory = BindBuffer(data.IndexBuffer, allocateInfo);

            var bytes = data.Indicies.SelectMany(BitConverter.GetBytes).ToArray(); // oh man, dat Linq tho
            CopyArrayToBuffer(data.IndexDeviceMemory, memorySize, bytes);

            data.BindingDescriptions = new[]
            {
                new VertexInputBindingDescription(0, (uint)(sizeof(float) * quadVertices.GetLength(1)), VertexInputRate.Vertex)
            };

            data.AttributeDescriptions = new[]
            {
                new VertexInputAttributeDescription(0, 0, Format.R32G32B32A32_SFLOAT, 0),                 // Vertex: X, Y, Z
                new VertexInputAttributeDescription(1, 0, Format.R32G32B32_SFLOAT, sizeof(float) * 3),    // UV: U, V
                new VertexInputAttributeDescription(2, 0, Format.R32G32B32A32_SFLOAT, sizeof(float) * 5), // Normal: X, Y, Z
            };

            return data;
        }

        ImageData LoadTexture(string filename, Queue queue, CommandPool cmdPool)
        {
            //
            var bmp = new Bitmap(filename);

            var bitmapFormat = System.Drawing.Imaging.PixelFormat.Format32bppArgb;
            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var bitmapData = bmp.LockBits(rect, System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmapFormat);
            //

            uint imageWidth  = (uint)bmp.Width;
            uint imageHeight = (uint)bmp.Height;

            var imageFormat   = Format.B8G8R8A8_UNORM;
            var imageData     = new ImageData();
            imageData.Width = imageWidth;
            imageData.Height = imageHeight;
            imageData.Image = CreateTextureImage(imageFormat, imageWidth, imageHeight);
            imageData.Memory = BindImage(imageData.Image);
            imageData.View = CreateImageView(imageData.Image, imageFormat);
            imageData.Sampler = CreateSampler();

            var memRequirements = device.GetImageMemoryRequirements(imageData.Image);
            var imageBuffer = CreateBuffer(memRequirements.Size, BufferUsageFlags.TransferSrc | BufferUsageFlags.TransferDst);
            var memoryIndex = FindMemoryIndex(MemoryPropertyFlags.HostVisible);
            var memAlloc = new MemoryAllocateInfo(memRequirements.Size, memoryIndex);
            var bufferMemory = BindBuffer(imageBuffer, memAlloc);

            CopyBitmapToBuffer(bitmapData.Scan0, (int)(imageWidth * imageHeight * 4), bufferMemory, memRequirements.Size);

            //
            var cmdBuffers = AllocateCommandBuffers(cmdPool, 1);
            var cmdBuffer = cmdBuffers[0];

            var beginInfo = new CommandBufferBeginInfo();
            cmdBuffer.Begin(beginInfo);

            PipelineBarrierSetLayout(cmdBuffer, imageData.Image, ImageLayout.Preinitialized, ImageLayout.TransferDstOptimal, AccessFlags.HostWrite, AccessFlags.TransferWrite);
            CopyBufferToImage(cmdBuffer, imageData, imageBuffer);
            PipelineBarrierSetLayout(cmdBuffer, imageData.Image, ImageLayout.TransferDstOptimal, ImageLayout.ShaderReadOnlyOptimal, AccessFlags.TransferWrite, AccessFlags.ShaderRead);

            // wait... why does this work?
            device.DestroyBuffer(imageBuffer);
            device.FreeMemory(bufferMemory);

            cmdBuffer.End();

            var submitInfo = new SubmitInfo(null, null, new[]{ cmdBuffer }, null);
            queue.Submit(new[] { submitInfo });
            submitInfo.Dispose();
            queue.WaitIdle();

            device.FreeCommandBuffers(cmdPool, cmdBuffer);
            //

            //CopyBufferToImage(queue, cmdPool, imageData, imageBuffer);

            //
            bmp.UnlockBits(bitmapData);
            bmp.Dispose();
            //

            return imageData;
        }

        Format GetDepthStencilFormat(PhysicalDevice physDevice)
        {
            var depthFormats = new[]
            {
                Format.D32_SFLOAT_S8_UINT,
                Format.D32_SFLOAT,
                Format.D24_UNORM_S8_UINT,
                Format.D16_UNORM_S8_UINT,
                Format.D16_UNORM
            };

            foreach(var f in depthFormats)
            {
                var properties = physDevice.GetFormatProperties(f);
                if(properties.OptimalTilingFeatures.HasFlag(FormatFeatureFlags.DepthStencilAttachment))
                    return f;
            }

            throw new Exception("Found no valid depth stencil format!");
        }

        ImageData CreateDepthStencil(Format stencilFormat, uint width, uint height)
        {
            var size = new Extent3D(width, height, 1);
            var usage = ImageUsageFlags.DepthStencilAttachment | ImageUsageFlags.TransferSrc;
            var createImageInfo = new ImageCreateInfo(ImageType.ImageType2d, stencilFormat, size, 1, 1, SampleCountFlags.SampleCountFlags1, ImageTiling.Optimal, usage, SharingMode.Exclusive, null, ImageLayout.Preinitialized);
            var img = device.CreateImage(createImageInfo);

            var imgMem = BindImage(img);

            var subresourceRange = new ImageSubresourceRange(ImageAspectFlags.Depth | ImageAspectFlags.Stencil, 0, 1, 0, 1);
            var createInfo = new ImageViewCreateInfo(img, ImageViewType.ImageViewType2d, stencilFormat, new ComponentMapping(), subresourceRange);
            var imgView = device.CreateImageView(createInfo);

            return new ImageData
            {
                Height = height,
                Width = width,
                Image = img,
                Memory = imgMem,
                View = imgView
            };
        }

        void SetStencilLayout(CommandPool cmdPool, Queue queue, ImageData stencil)
        {
            var allocInfo = new CommandBufferAllocateInfo(cmdPool, CommandBufferLevel.Primary, 1);
            var cmdBuffer = device.AllocateCommandBuffers(allocInfo).First();

            var beginInfo = new CommandBufferBeginInfo();
            cmdBuffer.Begin(beginInfo);

            var subresourceRange = new ImageSubresourceRange(ImageAspectFlags.Depth | ImageAspectFlags.Stencil, 0, 1, 0, 1);
            var imageMemoryBarrier = new ImageMemoryBarrier(ImageLayout.Undefined, ImageLayout.DepthStencilAttachmentOptimal, 0, 0, stencil.Image, subresourceRange);
            imageMemoryBarrier.SrcAccessMask = AccessFlags.None;
            imageMemoryBarrier.DstAccessMask = AccessFlags.DepthStencilAttachmentWrite;
            var imageMemoryBarriers = new[]{ imageMemoryBarrier };
            cmdBuffer.PipelineBarrier(PipelineStageFlags.TopOfPipe, PipelineStageFlags.TopOfPipe, DependencyFlags.None, null, null, imageMemoryBarriers);
            imageMemoryBarrier.Dispose();

            cmdBuffer.End();

            var submitInfo = new SubmitInfo(null, null, new[]{ cmdBuffer }, null);
            queue.Submit(new[] { submitInfo });
            submitInfo.Dispose();

            queue.WaitIdle();
            device.WaitIdle();
        }

        #endregion

        #region  Primary Initialization
        
        protected Instance CreateInstance(string[] enabledLayers, string[] enabledExtensions)
        {
            using(var instanceCreateInfo = new InstanceCreateInfo(enabledLayers, enabledExtensions))
                return Vulkan.Managed.Vk.CreateInstance(instanceCreateInfo);
        }

        protected PhysicalDevice[] EnumeratePhysicalDevices(Instance instance)
        {
            var physicalDevices = instance.EnumeratePhysicalDevices();

            if(physicalDevices.Length == 0)
                throw new InvalidOperationException("Didn't find any physical devices");

            return physicalDevices;
        }

        protected Device CreateDevice(PhysicalDevice physicalDevice, uint queueFamily, string[] enabledLayers, string[] enabledExtensions)
        {
            var features = new PhysicalDeviceFeatures();
            features.ShaderClipDistance = true;
            features.ShaderCullDistance = true;
            
            var queueCreateInfo = new DeviceQueueCreateInfo(queueFamily, new[]{ 0f });
            using(var deviceCreateInfo = new DeviceCreateInfo(new[] { queueCreateInfo }, enabledLayers, enabledExtensions))
            {
                deviceCreateInfo.EnabledFeatures = features;
                return physicalDevice.CreateDevice(deviceCreateInfo);
            }
        }

        protected Queue GetQueue(PhysicalDevice physicalDevice, uint queueFamily)
        {
            var queueNodeIndex = physicalDevice.GetQueueFamilyProperties()
                .Where((p, i) => (p.QueueFlags & QueueFlags.Graphics) != 0)
                .Select((p, i) => i)
                .First();

            return device.GetQueue(queueFamily, (uint)queueNodeIndex);
        }

        protected CommandPool CreateCommandPool(uint queueFamily)
        {
            using(var commandPoolCreateInfo = new CommandPoolCreateInfo(queueFamily))
            {
                commandPoolCreateInfo.Flags = CommandPoolCreateFlags.ResetCommandBuffer;
                return device.CreateCommandPool(commandPoolCreateInfo);
            }
        }

        protected CommandBuffer[] AllocateCommandBuffers(CommandPool commandPool, uint buffersToAllocate)
        {
            using(var commandBufferAllocationInfo = new CommandBufferAllocateInfo(commandPool, CommandBufferLevel.Primary, buffersToAllocate))
            {
                var commandBuffers = device.AllocateCommandBuffers(commandBufferAllocationInfo);

                if(commandBuffers.Length == 0 || commandBuffers.Length != buffersToAllocate)
                    throw new InvalidOperationException($"Expected to allocate {buffersToAllocate} command buffers, got {commandBuffers.Length} instead!");

                return commandBuffers;
            }
        }

        #endregion

        #region Surface / Swapchain
        
        protected SwapchainData CreateSwapchain(PhysicalDevice physicalDevice, SurfaceKHR surface, uint windowWidth, uint windowHeight)
        {
            var data = new SwapchainData();

            var surfaceFormats = physicalDevice.GetSurfaceFormatsKHR(surface);
            var surfaceFormat  = surfaceFormats[0].Format;

            var surfaceCapabilities = physicalDevice.GetSurfaceCapabilitiesKHR(surface);

            var desiredImageCount = Math.Min(surfaceCapabilities.MinImageCount + 1, surfaceCapabilities.MaxImageCount);

            SurfaceTransformFlagsKHR preTransform;
            if((surfaceCapabilities.SupportedTransforms & SurfaceTransformFlagsKHR.Identity) != 0)
            {
                preTransform = SurfaceTransformFlagsKHR.Identity;
            }
            else
            {
                preTransform = surfaceCapabilities.CurrentTransform;
            }

            var presentModes = physicalDevice.GetSurfacePresentModesKHR(surface);

            var swapChainPresentMode = PresentModeKHR.Fifo;
            if(presentModes.Contains(PresentModeKHR.Mailbox))
                swapChainPresentMode = PresentModeKHR.Mailbox;
            else if(presentModes.Contains(PresentModeKHR.Immediate))
                swapChainPresentMode = PresentModeKHR.Immediate;

            var imageExtent = new Extent2D(windowWidth, windowHeight);
            var swapchainCreateInfo = new SwapchainCreateInfoKHR
            {
                Surface            = surface,
                MinImageCount      = desiredImageCount,

                ImageFormat        = surfaceFormat,
                ImageExtent        = imageExtent,
                ImageArrayLayers   = 1,
                ImageUsage         = ImageUsageFlags.ColorAttachment,
                ImageSharingMode   = SharingMode.Exclusive,
                //ImageColorSpace    = ColorSpaceKHR.SrgbNonlinear,

                QueueFamilyIndices = null,
                PreTransform       = preTransform,
                CompositeAlpha     = CompositeAlphaFlagsKHR.Opaque,
                PresentMode        = swapChainPresentMode,
                Clipped            = true,
            };
            data.Swapchain = device.CreateSwapchainKHR(swapchainCreateInfo);
            data.ImageFormat = surfaceFormat;

            return data;
        }

        protected List<ImageData> InitializeSwapchainImages(Queue queue, CommandPool cmdPool, Image[] images, Format imageFormat, uint width, uint height)
        {
            var imageDatas = new List<ImageData>();

            foreach(var img in images)
            {
                var imgData    = new ImageData();
                imgData.Image = img;
                imgData.Width = width;
                imgData.Height = height;
                imgData.View = CreateImageView(img, imageFormat);
                imageDatas.Add(imgData);
            }

            return imageDatas;
        }

        #endregion

        #region Dependencies

        Image CreateImage(Format imageFormat, uint width, uint height)
        {
            var size = new Extent3D(width, height, 1);
            var usage = ImageUsageFlags.ColorAttachment | ImageUsageFlags.TransferSrc | ImageUsageFlags.TransferDst;
            var createImageInfo = new ImageCreateInfo(ImageType.ImageType2d, imageFormat, size, 1, 1, SampleCountFlags.SampleCountFlags1, ImageTiling.Optimal, usage, SharingMode.Exclusive, null, ImageLayout.Preinitialized);
            return device.CreateImage(createImageInfo);
        }

        Image CreateTextureImage(Format imageFormat, uint width, uint height)
        {
            var size = new Extent3D(width, height, 1);
            var usage = ImageUsageFlags.TransferDst | ImageUsageFlags.Sampled;
            var createImageInfo = new ImageCreateInfo(ImageType.ImageType2d, imageFormat, size, 1, 1, SampleCountFlags.SampleCountFlags1, ImageTiling.Optimal, usage, SharingMode.Exclusive, null, ImageLayout.Preinitialized);
            return device.CreateImage(createImageInfo);
        }

        DeviceMemory BindImage(Image image)
        {
            var memRequirements = device.GetImageMemoryRequirements(image);
            var memTypeIndex = FindMemoryIndex(MemoryPropertyFlags.DeviceLocal);
            var memAlloc = new MemoryAllocateInfo(memRequirements.Size, memTypeIndex);
            var deviceMem = device.AllocateMemory(memAlloc);
            device.BindImageMemory(image, deviceMem, 0);
            return deviceMem;
        }

        ImageView CreateImageView(Image image, Format imageFormat)
        {
            var subresourceRange = new ImageSubresourceRange(ImageAspectFlags.Color, 0, 1, 0, 1);
            var createInfo = new ImageViewCreateInfo(image, ImageViewType.ImageViewType2d, imageFormat, new ComponentMapping(), subresourceRange);
            return device.CreateImageView(createInfo);
        }

        RenderPass CreateRenderPass(Format imageFormat)
        {
            var imageLayout = ImageLayout.ColorAttachmentOptimal;

            var attachmentDescriptions = new[]
            {
                new AttachmentDescription
                {
                    Format         = imageFormat,
                    Samples        = SampleCountFlags.SampleCountFlags1,
                    StencilLoadOp  = AttachmentLoadOp.DontCare,
                    StencilStoreOp = AttachmentStoreOp.DontCare,
                    InitialLayout  = imageLayout,
                    FinalLayout    = imageLayout
                },
            };

            var colorAttachmentReferences = new[]
            {
                new AttachmentReference(0, imageLayout)
            };

            var subpassDescriptions = new[]
            {
                new SubpassDescription(PipelineBindPoint.Graphics, null, colorAttachmentReferences, null)
            };

            var createInfo = new RenderPassCreateInfo(attachmentDescriptions, subpassDescriptions, null);
            return device.CreateRenderPass(createInfo);
        }

        RenderPass CreateRenderPass(Format colorFormat, Format stencilFormat)
        {
            var attachmentDescriptions = new[]
            {
                new AttachmentDescription
                {
                    Format         = colorFormat,
                    Samples        = SampleCountFlags.SampleCountFlags1,
                    StencilLoadOp  = AttachmentLoadOp.DontCare,
                    StencilStoreOp = AttachmentStoreOp.DontCare,
                    InitialLayout  = ImageLayout.ColorAttachmentOptimal,
                    FinalLayout    = ImageLayout.ColorAttachmentOptimal
                },
                new AttachmentDescription
                {
                    Format         = stencilFormat,
                    Samples        = SampleCountFlags.SampleCountFlags1,
                    StencilLoadOp  = AttachmentLoadOp.DontCare,
                    StencilStoreOp = AttachmentStoreOp.DontCare,
                    InitialLayout  = ImageLayout.DepthStencilAttachmentOptimal,
                    FinalLayout    = ImageLayout.DepthStencilAttachmentOptimal
                },
            };

            var colorAttachmentReferences = new[]
            {
                new AttachmentReference(0, ImageLayout.ColorAttachmentOptimal)
            };

            var subpassDescriptions = new[]
            {
                new SubpassDescription(PipelineBindPoint.Graphics, null, colorAttachmentReferences, null)
                {
                    DepthStencilAttachment = new AttachmentReference(1, ImageLayout.DepthStencilAttachmentOptimal),
                }
            };

            var createInfo = new RenderPassCreateInfo(attachmentDescriptions, subpassDescriptions, null);
            return device.CreateRenderPass(createInfo);
        }

        PipelineShaderStageCreateInfo GetShaderStageCreateInfo(ShaderStageFlags stage, string filename, string entrypoint = "main")
        {
            var shaderBytes = File.ReadAllBytes(filename);
            return new PipelineShaderStageCreateInfo(stage, CreateShaderModule(shaderBytes), entrypoint);
        }

        ShaderModule CreateShaderModule(byte[] shaderCode)
        {
            var createInfo = new ShaderModuleCreateInfo(shaderCode);
            return device.CreateShaderModule(createInfo);
        }

        PipelineLayout CreatePipelineLayout(DescriptorSetLayout descriptorSetLayout)
        {
            var createInfo = new PipelineLayoutCreateInfo(new[]{ descriptorSetLayout }, null);
            return device.CreatePipelineLayout(createInfo);
        }

        Pipeline[] CreatePipelines(PipelineLayout pipelineLayout, RenderPass renderPass, PipelineShaderStageCreateInfo[] shaderStageCreateInfos, VertexData vertexData)
        {
            var inputAssemblyState = new PipelineInputAssemblyStateCreateInfo(PrimitiveTopology.TriangleList, false);

            var vertexInputState = new PipelineVertexInputStateCreateInfo(vertexData.BindingDescriptions, vertexData.AttributeDescriptions);

            var rasterizationState = new PipelineRasterizationStateCreateInfo();
            rasterizationState.LineWidth = 1;

            //var blendState = new PipelineColorBlendStateCreateInfo();

            //var viewportState = new PipelineViewportStateCreateInfo();
            //viewportState.Viewports = 1;
            //viewportState.Scissors = 1;

            var depthStencilState = new PipelineDepthStencilStateCreateInfo();
            depthStencilState.DepthTestEnable = true;
            depthStencilState.DepthWriteEnable = true;
            depthStencilState.DepthCompareOp = CompareOp.LessOrEqual;
            depthStencilState.Back = new StencilOpState { CompareOp = CompareOp.Always };
            depthStencilState.Front = new StencilOpState { CompareOp = CompareOp.Always };

            var createInfos = new[]
            {
                new GraphicsPipelineCreateInfo(shaderStageCreateInfos, vertexInputState, inputAssemblyState, rasterizationState, pipelineLayout, renderPass, 0, 0)
                {
                    ViewportState = new PipelineViewportStateCreateInfo(),
                    MultisampleState = new PipelineMultisampleStateCreateInfo()
                    {
                        RasterizationSamples = SampleCountFlags.SampleCountFlags1
                    },
                    DepthStencilState = depthStencilState,
                }
            };

            return device.CreateGraphicsPipelines(null, createInfos);
        }

        Framebuffer CreateFramebuffer(RenderPass renderPass, params ImageData[] imageData)
        {
            var attachments = imageData.Select(x => x.View).ToArray();
            var createInfo = new FramebufferCreateInfo(renderPass, attachments, imageData[0].Width, imageData[0].Height, 1);
            return device.CreateFramebuffer(createInfo);
        }

        Buffer CreateBuffer(DeviceSize size, BufferUsageFlags flags)
        {
            var bufferCreateInfo = new BufferCreateInfo(size, flags, SharingMode.Exclusive, null);
            return device.CreateBuffer(bufferCreateInfo);
        }

        DeviceMemory BindBuffer(Buffer buffer, MemoryAllocateInfo allocInfo)
        {
            var bufferMem = device.AllocateMemory(allocInfo);
            device.BindBufferMemory(buffer, bufferMem, 0);
            return bufferMem;
        }
        
        Sampler CreateSampler()
        {
            var createInfo = new SamplerCreateInfo();
            createInfo.MagFilter = Filter.Linear;
            createInfo.MinFilter = Filter.Linear;
            createInfo.MipmapMode = SamplerMipmapMode.Linear;
            createInfo.MaxLod = 1;
            createInfo.AnisotropyEnable = true;
            createInfo.BorderColor = BorderColor.FloatOpaqueWhite;
            return device.CreateSampler(createInfo);
        }

        UniformData CreateUniformBuffer(Type targetType)
        {
            var data = new UniformData();
            var size = Marshal.SizeOf(targetType);
            data.Buffer = CreateBuffer((ulong)size, BufferUsageFlags.UniformBuffer);

            var memRequirements = device.GetBufferMemoryRequirements(data.Buffer);
            var memoryIndex = FindMemoryIndex(MemoryPropertyFlags.HostVisible);
            var memAlloc = new MemoryAllocateInfo(memRequirements.Size, memoryIndex);
            data.Memory = BindBuffer(data.Buffer, memAlloc);

            data.Descriptor = new DescriptorBufferInfo(data.Buffer, 0, (ulong)size);

            data.AllocSize = (uint)memAlloc.AllocationSize;

            return data;
        }

        Transform UpdateTransform(Transform ubo, float width, float height, float zoom)
        {
            ubo.projection = Matrix4.CreatePerspectiveFieldOfView(DegreesToRadians(60), width / height, 0.001f, 256.0f);
            ubo.view = Matrix4.LookAt(new Vector3(zoom, -3, -3), Vector3.Zero, Vector3.UnitY);
            ubo.model = Matrix4.Identity;
            return ubo;
        }

        void CopyTransform(Transform ubo, UniformData uniform)
        {
            var map = device.MapMemory(uniform.Memory, 0, uniform.AllocSize);

            var size = Marshal.SizeOf(typeof(Transform));
            var bytes = new byte[size];
            IntPtr ptr = Marshal.AllocHGlobal(size);
            Marshal.StructureToPtr(ubo, ptr, true);
            Marshal.Copy(ptr, bytes, 0, size);
            Marshal.FreeHGlobal(ptr);

            Marshal.Copy(bytes, 0, map, size);

            device.UnmapMemory(uniform.Memory);
        }

        DescriptorPool CreateDescriptorPool()
        {
            var poolSizes = new[]
            {
                new DescriptorPoolSize(DescriptorType.UniformBuffer, 1),
                new DescriptorPoolSize(DescriptorType.CombinedImageSampler, 1),
            };

            var createInfo = new DescriptorPoolCreateInfo(2, poolSizes);
            createInfo.Flags = DescriptorPoolCreateFlags.FreeDescriptorSet;
            return device.CreateDescriptorPool(createInfo);
        }

        DescriptorSetLayout CreateDescriptorSetLayout()
        {
            var layoutBindings = new[]
            {
                new DescriptorSetLayoutBinding(0, DescriptorType.UniformBuffer, ShaderStageFlags.Vertex),
                new DescriptorSetLayoutBinding(1, DescriptorType.CombinedImageSampler, ShaderStageFlags.Fragment),
            };
            layoutBindings[0].DescriptorCount = 1;
            layoutBindings[1].DescriptorCount = 1;

            var createInfo = new DescriptorSetLayoutCreateInfo(layoutBindings);
            return device.CreateDescriptorSetLayout(createInfo);
        }

        DescriptorSet CreateDescriptorSet(DescriptorPool pool, DescriptorSetLayout layout, ImageData imageData, UniformData uniformData)
        {
            var allocInfo = new DescriptorSetAllocateInfo(pool, new[] { layout });
            var sets = device.AllocateDescriptorSets(allocInfo);
            var descriptorSet = sets.First();
            var texDescriptor = new DescriptorImageInfo(imageData.Sampler, imageData.View, ImageLayout.General);
            var writeDescriptorSets = new[]
            {
                new WriteDescriptorSet(descriptorSet, 0, 0, DescriptorType.UniformBuffer, null, null, null),
                new WriteDescriptorSet(descriptorSet, 1, 0, DescriptorType.CombinedImageSampler, null, null, null),
            };
            writeDescriptorSets[0].BufferInfo = new[] { uniformData.Descriptor };
            writeDescriptorSets[1].ImageInfo = new[] { texDescriptor };
            device.UpdateDescriptorSets(writeDescriptorSets, null);
            return descriptorSet;
        }

        #endregion

        void SubmitForExecution(Queue queue, Semaphore presentSemaphore, CommandBuffer cmdBuffer)
        {
            var submitInfo = new SubmitInfo(new[]{ presentSemaphore }, null, new[]{ cmdBuffer }, null);
            queue.Submit(new[] { submitInfo });
            submitInfo.Dispose();
        }

        void PipelineBarrierSetLayout(CommandBuffer cmdBuffer, Image image, ImageLayout oldLayout, ImageLayout newLayout, AccessFlags srcMask, AccessFlags dstMask)
        {
            var subresourceRange = new ImageSubresourceRange(ImageAspectFlags.Color, 0, 1, 0, 1);
            var imageMemoryBarrier = new ImageMemoryBarrier(oldLayout, newLayout, 0, 0, image, subresourceRange);
            imageMemoryBarrier.SrcAccessMask = srcMask;
            imageMemoryBarrier.DstAccessMask = dstMask;
            var imageMemoryBarriers = new[]{ imageMemoryBarrier };
            cmdBuffer.PipelineBarrier(PipelineStageFlags.TopOfPipe, PipelineStageFlags.TopOfPipe, DependencyFlags.None, null, null, imageMemoryBarriers);
            imageMemoryBarrier.Dispose();
        }

        #region Copy

        void CopyImageToBuffer(CommandBuffer cmdBuffer, ImageData imageData, Buffer imageBuffer, uint width, uint height)
        {
            var subresource = new ImageSubresourceLayers(ImageAspectFlags.Color, 0, 0, 1);
            var imageCopy = new BufferImageCopy(0, width, height, subresource, new Offset3D(0, 0, 0), new Extent3D(width, height, 0));
            cmdBuffer.CopyImageToBuffer(imageData.Image, ImageLayout.TransferSrcOptimal, imageBuffer, imageCopy);
        }

        void CopyBufferToImage(CommandBuffer cmdBuffer, ImageData imageData, Buffer imageBuffer)
        {
            var subresource = new ImageSubresourceLayers(ImageAspectFlags.Color, 0, 0, 1);
            var imageCopy = new BufferImageCopy(0, 0, 0, subresource, new Offset3D(0, 0, 0), new Extent3D(imageData.Width, imageData.Height, 1));
            cmdBuffer.CopyBufferToImage(imageBuffer, imageData.Image, ImageLayout.TransferDstOptimal, imageCopy);
        }

        void CopyArrayToBuffer(DeviceMemory bufferMem, DeviceSize size, byte[] data)
        {
            var map = device.MapMemory(bufferMem, 0, size);
            Marshal.Copy(data, 0, map, (int)((ulong)size));
            device.UnmapMemory(bufferMem);
        }

        void CopyBitmapToBuffer(IntPtr scan0, int bitmapSize, DeviceMemory bufferMem, DeviceSize size)
        {
            var map = device.MapMemory(bufferMem, 0, size);
            Copy(scan0, map, bitmapSize);
            device.UnmapMemory(bufferMem);
        }

        void Copy(IntPtr src, IntPtr dest, int size)
        {
            var data = new byte[size];
            Marshal.Copy(src, data, 0, size);
            Marshal.Copy(data, 0, dest, size);
        }

        #endregion

        uint FindMemoryIndex(MemoryPropertyFlags propertyFlags)
        {
            for(uint x = 0; x < VulkanConstant.MaxMemoryTypes; x++)
                if((physDeviceMem.MemoryTypes[x].PropertyFlags & propertyFlags) == propertyFlags)
                    return x;

            throw new InvalidOperationException();
        }

        float DegreesToRadians(float degrees)
        {
            const float degToRad = (float)Math.PI / 180.0f;
            return degrees * degToRad;
        }

        private Bool32 DebugReport(DebugReportFlagsEXT flags, DebugReportObjectTypeEXT objectType, ulong @object, IntPtr location, int messageCode, string layerPrefix, string message, IntPtr userData)
        {
            if(messageCode != 0)
                Console.WriteLine($"[{messageCode,2}] {message}");
            return false;
        }

        #endregion
    }
}
