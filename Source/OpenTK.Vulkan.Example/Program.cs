using System;
using System.Drawing;

using OpenTK;
using OpenTK.Graphics;
using OpenTK.Input;

using Vulkan;

namespace OpenTK.VulkanExample
{
    class VulkanExample
    {
        public static void Main()
        {
            // CONTROLS
            // W A S D - Move (World Coordinates)
            // Z X - Rotate around Y axis (left/right)
            // F - Print position to console
            // R - Set position to (0,0,0)
            var cubeExample = new RotatingCubeExample();
            cubeExample.Run();
            Console.WriteLine("program complete (press any key)");
            Console.ReadKey();
        }
    }

    public interface ISurfaceProvider
    {
        SurfaceKHR CreateSurface(Instance instance);
    }

    public class Win32SurfaceProvider : ISurfaceProvider
    {
        OpenTK.Platform.Windows.WinGLNative _window;

        public Win32SurfaceProvider(OpenTK.Platform.Windows.WinGLNative window)
        {
            _window = window;
        }

        public SurfaceKHR CreateSurface(Instance instance)
        {
            var createInfo = new Vulkan.Managed.Win32SurfaceCreateInfoKHR(_window.Vulkan_InstancePtr, _window.Vulkan_WndPtr);
            return Vulkan.Managed.Vk.CreateWin32SurfaceKHR(instance, createInfo);
        }
    }
}