# Structure

- Instance:
  - First Object Created for app
  - Way to create other API Objects

- Physical Device
  - Reference to gpu
  - Has extensions that may or may not be present

- Logical Device
  - Interface to the physical device gpu
  - Used to create most other fundamental objects

- Queue Families
  - Each gpu has queues that originate from certain queue families
    - Graphics, Compute, Present, Etc
  - Queues can have multiple families
  - All actions in Vulkan are submitted through queues

- Surface
  - Interface between Vulkan and the platform window
  - Uses Presentation queue

## Swapchain Bullshit

- 