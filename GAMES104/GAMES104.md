# GAMES104 - Modern Game Engine

> Xi Wang | [Live](https://www.huya.com/19077762) | [Site](games104.boomingtech.com) | [BBS](https://games-cn.org/forums/forum/games104-forum/) 
>
> Ref: Jason Gregroy, “*Game Engine Architecture*”, 3rd/4th edition



# Lecture 1 Overview of Game Engine 

## Definition of Game Engine

- Technology Foundation of Matrix
- Productivity Tools of Creation
- Art of Complexity

## Developer Platform

- For programmer

  Expandable API interface allow programmers to define various of gameplay without changing the core

- For studio

  Collaborate hundreds of developers with different disciplinary work smoothly together

## Lecture Layout

- **Basic Elements**

  - Engine structure and layer
  - Data organization and management

- **Rendering**

  - Model, material, shader, texture
  - Light and shadow
  - Render pipeline
  - Sky, terrain, etc.

- **Animation**

  - Basic concepts of animation
  - Animation structure and pipeline

- **Physics**

  - Basic concepts of physics system
  - Gameplay application
  - Performance optimization

- **Gameplay**

  Event system / Scripts system / Graph Driven

- **Misc System**

  Effects / Navigation / Camera / …

- **Tool Set**

  - C++ Reflection
  - Data schema

- **Online Gaming**

  - Lockstep synchronization
  - State synchronization
  - Consistency

- **Advanced Technology**

  - Motion matching
  - Procedural content generation (PCG)
  - Data-oriented programming (DOP)
  - Job system
  - Lumen (dynamic GI and reflections)
  - Nanite



# Lecture 2 Layered Architecture of Game Engine

## 5 Layers

![image-20220323002341833](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220323002341833.png)

- Tool Layer
- Function Layer
- Resource Layer
- Core Layer
- Platform Layer
- 3rd party libraries*

## Resource Layer

### To Access Data

> data and files: scenes and level script and graph game logic data

**Offline Resource Importing** (resource -> **asset** (high efficiency in engine))

- Unify file access by defining a meta asset file format (ie. ast)
- Assets are faster to access by importing preprocess
- Build a composite asset file to refer to all resources
- GUID is an extra protection of reference  

**Runtime Resource Management**

- A virtual file system to load/unload assets by path reference
- Manage asset lifespan and reference by **handle** system

### Manage Asset Life Cycle

Memory management for resources - life cycle

- Different resources have different life cycles
- Limited memory requires release of loaded resources when possible
- Garbage collection and deferred loading is critical features

![image-20220323002719851](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220323002719851.png)

## Function Layer

> Make visible, movable and playable
>
> Physics, animation, rendering, camera, HUD and input, script, FSM and AI

### Tick

After a “tick” everything in the system is updated

In a `tick` function, there are 2 essential functions `tickLogic` and `tickRender` 

In a tick:

-  Fetch animation frame of character
-  Drive the skeleton and skin of character
-  Renderer process all rendering jobs in an iteration of render tick for each **frame**  

### Heavy-duty Hotchpotch

- Function Layer provides major function modules for the game engine

  - Object system (HUGE)
- Game Loop updates the systems periodically
  - Game Loop is the key of reading codes of game engines

- Blur the boundary between engine and game
    - Camera, character and behavior
    - Design extendable engine API for programmer  


### Multi-Threading

Multi-core processors become the mainstream

​	Fixed thread (entry) -> Thread fork/join (mainstream) -> JOB system (advanced)

Many systems in game engine are built for <u>parallelism</u>  

## Core Layer

> Swiss knife of game engine: called by various stuff (animation/physics/render/script/cam/…)

### Math Library

Linear algebra: translation / scaling / rotation; matrix splines / quaternion

Math efficiency (why use spec lib)

Hacks: Carmack’s `1/sqrt(x)` (very slow); magic number …

SIMD: Do 4 operations in one LU (SSE - parallelized matrix operations)

![image-20220323003143080](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220323003143080.png)

### Data Structure and Containers

- Vecotrs / maps / trees / …
- Customized outperforms STL
- Avoid <u>fragment</u> memory (-> not use the default C++ memory management)
- Skeleton tree
- Animation frame seq

### Memory Management

- Major bottlenecks of game engine performance
  - Memory pool / allocator
  - Reduce cache miss
  - Memory alignment
- Polymorphic Memory Resource (PMR)

###  Foundation of Game Engine

- Core layers provide utilities needed in various function modules
- Super high performance design and implementation
- High standard of coding

![image-20220323003248550](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220323003248550.png)

## Platform Layer

> Launch on different platforms: operation systems (Windows/MacOS/Linux/…), platform file sys, graphics API, Platform SDK, consoles, input devices, publishing platforms (Steam/XGP/EGS/…)
>

**File system**

- Path: Slash/backslash, Environment variables
- Directory Traversal  

**Graphics API**: **Render Hardware Interface (RHI)** 

- Transparent different GPU architectures and SDK
- Automatic optimization of target platforms

## Tool Layer

> not run-time

### Allow Anyone to Create Game 

- Unleash the creativity
  - Build upon game engine
  - Create, edit and exchange game play assets
- Flexible of coding languages

### Digital Content Creation (DCC)

Editors + DCC = Asset Conditioning Pipeline => Game | Game Engine (becomes assets)

## Layered Archetecture

- Decoupling and Reducing Complexity
  - Lower layers and independent from upper layers (core)
  - Upper layers don’t know how lower layers are implemented (tool/…)
- Response for Evolving Demands
  - Upper layers evolve fast, but lower layers are stable



# Lecture 3 Build a Game World

## Make the Game World 

### Game Objects

- **Dynamic objects** (interactive)
- **Static objects**
- **Environments** (sky/vegetation/terrain)
- **Other game objects** (air wall/trigger area/navigation mesh/ruler)

Everything is a <u>game object (GO)</u>

To describe a game object: <u>properties</u> (shape/position/health/capacity of battery), <u>behavior</u> (move/scout), etc. => a `class` in earliest game engines

> From a drone to a armed drone, can add “ammo”, “fire”, etc. => use parent classes (the <u>derivation</u> and <u>inheritance</u> of objects)
>
> ``` c#
> // Inheritance (OOP idea)
> class ArmedDrone:
> 	public Drone
> {
>     public:
>         float ammo;
>         void fire();
> }
> ```

But no perfect classification in the game world

### Component Base 

> For a drone: transform, motor, model, animation physics, AI, …

Use a base function and a lot of components 

> In UE, UObject is not exactly GO

Disadvantage: The efficiency is lower than directly classes (Messaging API)

### Ticks

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329175042542.png" alt="image-20220329175042542" style="zoom:67%;" />

**Object-based tick vs. Component-based tick**:

- Obj-based
  - Simple and intuitive
  - Easy to debug
- Comp-based (more efficient)
  - Parallelized processing
  - Reduced cache miss

> If a tick is too long: process obj by batches in up to 5 frames (acceptable for human eyes)
>
> If animation and physics affect each other, can use interpolation between animation and physics (AI)

### Events Mechanism

To code these ticks:

- <u>Hardcode</u> (`switch` - `case` - process something) 
- <u>**Events**</u> (`sendEvent(go_id)`) 
  - Message sending and handling
  - Decoupling event sending and handling

> In commercial engines
>
> - In Unity: `gameObject.SendMessage("some string", n);`
> - In UE: `DECLARE_EVENT_SomeParam(...)`

Debugging: print log; visualization in 3D space (/blueprints)

## Manage Game Objects

### Scene Management

- Game obj are managed **in scene**
- Game obj **query**:
  - by <u>unique game obj ID</u>
  - by <u>obj position</u>

#### Simple space segmentation

No division: In small games with not that many obj, OK; but for big games with hundreds of thousands of obj, disaster (n^2^ problem)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329211426072.png" alt="image-20220329211426072" style="zoom:80%;" />

Problem: Divided by grid has a problem if the distribution not uniform (while in 3A games the dist of GO will not be uniform)

#### Hierarchical Segementation

By object clusters. For example: <u>Quadtree/Octree/BVH</u>

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329211806393.png" alt="image-20220329211806393" style="zoom:80%;" />

#### Spatial Data Structures

- Bounding Volume Hierarchies (BVH)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329214349292.png" alt="image-20220329214349292" style="zoom: 67%;" /> 

- Binary Space Partitioning (BSP)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329214442275.png" alt="image-20220329214442275" style="zoom:67%;" /> 

- Octree

  ![image-20220329214505200](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329214505200.png) 

- Scene Graph

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220329214515723.png" alt="image-20220329214515723" style="zoom:90%;" /> 

### Order of Messaging

- GO Bindings
- Component Dependencies (Animation <-> Physics <-> Motor -> Ani..)
- Immediate Event Sending or not

To keep the order in time is strictly -> <u>Lag</u>



# Lecture 4 Rendering on Game Engine

## Game Rendering

### Challenges

- Tens of thousands of objects with dozens type of effects

- Deal with architecture of modern computer with a complex combination of CPU and GPU (deep compative)

- Commit a bullet-proof framerate

  - 30FPS (60/120FPS + VR)
  - 1080P/4K/8K/…

- Limit access to CPU bandwidth and memory footprint

  Game logic, network, animation, physics and AI systems are major consumers of CPU and main memory

Rendering on Game Engine: a heavily optimized practical software framework to fulfill the critical rendering requirements of games on modern hardware (PC/consoles/mobiles)

### Outline of Rendering

1. **Basics of Game Rendering**
   - Hardware architecture
   - Render data organization
   - Visibility
2. **Materials, Shaders and Lighting**
   - PBR (materials: SG, MR)
   - Shader permutation
   - Lighting
     - Point/Directional lighting
     - IBL/Simple GI
3. **Special Rendering**
   - Terrain
   - Sky/Fog
   - Postprocess
4. **Pipeline**
   - Forward, derred rendering, forward plus
   - Real pipeline with mixed effects
   - Ring buffer and V-sync
   - Tiled-based rendering

> Not included: cartoon rendering; 2D rendering engine; subsurface; hair/fur

## Building Blocks of Rendering

### Rendering Pipeline and Data

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726164520820.png" style="zoom:67%;" />

### Computation 

- **Projection and Rasterization**

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404201742552.png" alt="image-20220404201742552" style="zoom:80%;" /> 

- **Shading**

  - Constants/Parameters
  - ALU algorithms
  - Texture sampling (expensive and complex)
  - Branches
  
- **Texture Sampling**

  1. Use two nearest mipmap levels
  2. Perform bilinear interpolation in both mip-maps
  3. Linearly interpolate between the results

## GPU

### SIMD and SIMT

- **SIMD (Single Instruction Multiple Data)**: Describes computers with multiple processing elem that perform the same operation on multiple data points simultaneously

  `SIMD_ADD c, a, b`

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404202354496.png" alt="image-20220404202354496" style="zoom:67%;" /> 

- **SIMT (Single Instruction Multiple Threads)**: An execution model used in parallel computing where single instruction, multiple data (SIMD) is combined with multithreading  

  `SIMT c, a, b` 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404202422200.png" alt="image-20220404202422200" style="zoom: 80%;" /> 

### GPU Architecture

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404202552322.png" alt="image-20220404202552322" style="zoom:80%;" />

- **GPC (Graphics Processing Cluster)**: a dedicated hardware block for computing, rasterization, shading and texturing
- **SM (Streaming Multiprocessor)**: part of the GPU that runs CUDA kernels
- **Texture Units**: A texture processing unit, that can fetch and filter a texture
- **CUDA Core**: parallel processor that allow data to be worked on simultaneously by different processors
- **Warp**: a collection of threads

### Data Flow from CPU to GPU

> von Neumann framework (sep of data and computations) -> bottleneck

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404202903244.png" alt="image-20220404202903244" style="zoom:67%;" />

- CPU and Main Memory
  - Data load/unload
  - Data preparation

- CPU to GPU
  - High latency
  - Limited bandwidth
- GPU and video memory
  - High performance parallel rendering

> Tips: Always minimize data transfer between CPU and GPU when possible

### Cache Efficiency

- Take full advantage of hardware parallel computing
- Try to avoid the von Neumann bottleneck

> Cache miss causes performance loss

### GPU Bounds and Performance

Application performance is limited by:

- Memory bounds
- ALU bounds
- TMU (Texture mapping unit) bound
- BW (Bandwidth) bound

## Renderable

### Mesh Render Component

- Everything is a game object in the game world
- Game object could be described in the componentbased way  

### Building Blocks of Renderable

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404203658620.png" alt="image-20220404203658620" style="zoom:67%;" />

### Mesh Primitive

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406230449852.png" alt="image-20220406230449852" style="zoom:50%;" />

### Vertex and Index Buffer

> Mesh premitive is not clear enough

- Vertex data
  - Vertex declaration
  - Vertex buffer
- Index data
  - Index declaration
  - Index buffer

![image-20220404204033939](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404204033939.png)

### Materials

Determine the appearance of objects, and how objects interact with light (differ from physics materials which determines the physics interactions)

Models: Phong model; PBR model; Subsurface material (Burley subsurface profile)

### Various Textures in Materials

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404204332323.png" alt="image-20220404204332098" style="zoom:50%;" />

### Variety of Shaders

> source codes as data

- Fix function shading shaders (such as Phong’s)
- Custom shaders

## Render Objects in Engine

### Coordinate System and Transformation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406230907176.png" alt="image-20220406230907176" style="zoom:67%;" />

Model assets are made based on local coordinate systems, and eventually render them 

### Object with many materials

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404204658105.png" alt="image-20220404204658105" style="zoom: 80%;" />

-> For one GO (single model) and apply many <u>submeshes</u> in the buffer (use offset) => to display different textures

but causes wasting of memory

### Resource Pool

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404204946373.png" alt="image-20220404204946373" style="zoom:67%;" />

### Instantiation 

Instance: Use Handle to Reuse Resouces

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404205119305.png" alt="image-20220404205119305" style="zoom:80%;" />

### Sort by Material 

Sorting same submeshes together

``` matlab
Initialize Resource Pools
Load Resources

Sort all Submeshes by Materials

for each Materials
	Update Parameters
	Update Textures
	Update Shader
	Update VertexBuffer
	Update IndexBuffer
	for each submeshes
		Draw Primitive
	end	
end
```

### GPU Batch Rendering

> Further step based on <u>Sort by Material</u> 

In modern games, many sub-objects in a scene are the same => don’t require to set for every single object but generate thousands of objects in one draw command (batch rendering) => Use GPU rather than CPU

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406231435163.png" alt="image-20220406231435163" style="zoom:80%;" />

## Visibility Culling

> The rendering is still not very efficient

Visibility culling is a very basic underlying system

For each view, there are a lot of objects which aren’t needed to be rendered (the terrian/particles/objects/…)

The basic idea of culling: test if the <u>bounding box</u> in the <u>view frustum</u>

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406232523772.png" alt="image-20220406232523772" style="zoom:67%;" />

### Simple Bound 

- Inexpensive intersection tests
- Tight fitting
- Inexpensive to compute
- Easy to rotate and transform
- Use little memory  

**Types**: Sphere; AABB (Axis-Aligned Bounding Box); OBB (Oriented Bounding Box); 8-DOP; Convex Hull; …

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406232559187.png" alt="image-20220406232559187" style="zoom: 33%;" />

### Hierarchical View Frustum Culling

- Quad Tree Culling

- BVH (Bounding Volume Hierarchy) Culling

  Construct and insertion of <u>BVH</u>: 

  - top-down

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406232955668.png" alt="image-20220406232955668" style="zoom:28%;" /> 

  - bottom-up

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406232939267.png" alt="image-20220406232939267" style="zoom: 33%;" /> 

  - incremental tree-instertion

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406232916253.png" alt="image-20220406232916253" style="zoom:30%;" />  

### PVS (Potential Visibility Set)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406233206814.png" alt="image-20220406233206814" style="zoom:50%;" />

> The *Portal* serie is an example of the usage of PVS (the view through the “portals”)

Determine potentially visible leaf nodes immediately from portal  

``` c
 for each portals
     getSamplePoints();
	for each portal faces
        for each leaf
            do ray casting between portal face and leaf
        end
    end             
end
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406233443774.png" alt="image-20220406233443774" style="zoom:50%;" />

**Pros**

- Much faster than BSP / Octree
- More flexible and compatible
- Preload resources by PVS  

### GPU Culling

Hardware solution -> highly parallelizable

Also can apply some hirerachy methods

Very useful for complex scenes

## Texture Compression

### Compressions

- Traditional image compression like JPG and PNG  
  - Good compression rates
  - Image quality
  - Designed to compress or decompress an entire image (not for part of them)

- In game texture compression
  - Decoding speed
  - Random access (Very important reason why not use JPG or PNG)
  - Compression rate and visual quality
  - Encoding speed  


Cannot use popular compression file formats to store textures (cannot easily query)

![image-20220404210635819](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404210635819.png)

### Block Compression  

Common block-based compression formats:

- On PC, BC7 (modern) or DXTC (old) formats  
- On mobile, ASTC (modern) or ETC / PVRTC (old) formats  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406235406841.png" alt="image-20220406235406841" style="zoom: 25%;" />

## Authoring Tools of Modeling

- **Polymodeling** (Max/Maya/Blender): flexible but heavy workload
- **Sculpting**: creative but large volume of data
- **Scanning**: realistic but large volume of data
- **Procedural modeling** (Houdini/Unreal): intelligent but hard to achieve

## Cluster-Based Mesh Pipeline

### Sculpting Tools Create Infinite Details

- Artists create models with infinite details
- From linear fps to open world fps, complex scene submit 10 more times triangles to GPU per-frame  

### Cluster-Based Mesh Pipeline  

- **GPU-Driven Rendering Pipeline** (2015)

  Mesh Cluster Rendering

  - Arbitrary number of meshes in single drawcall
  - GPU-culled by cluster bounds
  - Cluster depth sorting

- **Geometry Rendering Pipeline Architecture** (2021)

  Rendering primitives are divided as:

  - Batch: a single API draw (drawIndirect / drawIndexIndirect), composed of many Surfs  
  - Surf: submeshes based on materials, composed of many Clusters  
  - Cluster: 64 triangles strip  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406235743890.png" alt="image-20220406235743890" style="zoom: 33%;" />

### Programmable Mesh Pipeline

![image-20220404211621544](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404211621544.png)

### GPU Culling in Cluster-Based Mesh

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406235843485.png" alt="image-20220406235843485" style="zoom: 67%;" />

### Nanite

> UE5 new feature

Hierarchical LOD clusters with seamless boundary

Don’t need hardware support, but using a hierarchical cluster culling on the precomputed BVH tree by persistent threads (CS) on GPU instead of task shader  
