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

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404211621544.png" alt="image-20220404211621544" style="zoom:67%;" />

### GPU Culling in Cluster-Based Mesh

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220406235843485.png" alt="image-20220406235843485" style="zoom: 67%;" />

### Nanite

> UE5 new feature

Hierarchical LOD clusters with seamless boundary

Don’t need hardware support, but using a hierarchical cluster culling on the precomputed BVH tree by persistent threads (CS) on GPU instead of task shader  



# Lecture 5 Lighting, Materials and Shaders

> – Rendering 2



## Rendering Basis

### Participants of Rendering Computation

- **Lighting**: photon emit, bounce, absorb and perception is the origin of everything in rendering
- **Material**: how matter react to photon
- **Shader**: how to train and organize those micro-slaves to finish such a vast and dirty computation job between photon and materials

### The Rendering Equation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413230859103.png" alt="image-20220413230859103" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413230928569.png" alt="image-20220413230928569" style="zoom:50%;" />

-> but complex in real rendering

![](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325113950646.png)

### Three Main Challenges

#### 1a. Visibilitily to Lights

- Ray Casting Toward Light Source  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413231512039.png" alt="image-20220413231512039" style="zoom:50%;" /> 

- Shadow on and off

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413231534883.png" alt="image-20220413231534883" style="zoom:50%;" /> 

#### 1b. Light Source Complexity  

The differences between point sources and face sources

![image-20220413231741323](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413231741323.png)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413231713986.png" alt="image-20220413231713986" style="zoom:50%;" />

#### 2. Do Integral Efficiently on Hardware

- Brute-force way sampling
- Smarter sampling (Monte Carlo)
- Derive fast analytical solutions
  - Simplify the $f_r$:
    - Assumptions the optical properties of materials
    - Mathematical representation of materials
  - Simplify the $L_i$:
    - Deal with directional light, point light and spot light only
    - A mathematical representation of incident light sampling on a hemisphere, e.g., IBL and SH

#### 3. Any Matter will be Light Source

Direct vs. Indirect illumination

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413233012102.png" alt="image-20220413233012102" style="zoom: 50%;" />

#### Summary

1. How to get incoming radiance for any given incoming direction
2. Integral of lighting and scatting function on hemisphere is expensive
3. To evaluate incoming radiance, we have to compute yet another integral, i.e. rendering equation is recursive

## Starting from Simple

> Using simple light source. no radiosity, microfacet and BRDF, etc.

### Simple Light Solution

![image-20220413233617021](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413233617021.png)

- Using simple light source as main light
  - Directional light in most cases
  - Point and spot light in special case
- Using ambient light to hack others
  - A constant to represent mean of complex hemisphere irradiance
- Supported in graphics API  

``` glsl
glLightfv(GL_LIGHTO, GL_AMBIENT, light_ambient);
glLightfv(GL_LIGHTO, GL_DIFFUSE, light_diffuse);
glLightfv(GL_LIGHTO, GL_SPECULAR, light_specular);
glLightfv(GL_LIGHTO, GL_POSITION, light_position);
```

### Environment Map Reflection

> Early stage exploration of image-based lighting

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220413233756620.png" alt="image-20220413233756620" style="zoom:67%;" />

- Using env map to enhance glossary surface reflection
- Using env mipmap to represent roughness of surface

``` glsl
void main(){
    vec3 N = normalize(normal);
    vec3 V = normalize(camera_position - world_position);
    
    vec3 R = reflect(V, N);
    
    FragColor = texture(cube_texture, R);
}
```

### Math Behind Light Combo

- **Main light**: dominant light
- **Ambient light**: low-freq of irradiance sphere distribution
- **Env map**: high-freq of irradiance sphere distribution

### Blinn-Phong Materials

Light addition theory

![](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210722003122186.png)
$$
L = L_a + L_d + L_s = k_aI_a + k_d(I/r^2) \max(0,\vb{n\cdot l}) + k_s (I/r^2)\max(0,\vb{n\cdot h}) ^p
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414004115737.png" alt="image-20220414004115737" style="zoom:67%;" />

``` glsl
// set material ambient
glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, Ka);
// set material diffuse
glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Kd);
// set material specular
glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Ks);
```

**Problems**:

- <u>Not energy conservative</u> (unstable in ray-tracing) -> a lot of noise compare real energy conservative model 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414004326136.png" alt="image-20220414004326136" style="zoom:50%;" /> 

- Hard to model <u>complex realistic material</u> (not related to real material) -> plastic feeling 

  (traditional shading vs. PBR shading)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414004400919.png" alt="image-20220414004400919" style="zoom:50%;" /> 

### Shadow

Shadow is nothing but space when the light is blocked by an opaque object

Already obsolete method:

- Planar shadow
- Shadow volume 
- Projective texture

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414004433161.png" alt="image-20220414004433161" style="zoom:50%;" />

#### Shadow Mapping

In game engines -> **Shadow map**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411202713487.png" alt="image-20220411202713487" style="zoom:67%;" />

``` glsl
// project our 3D position to the shadow map
vec4 proj_pos = shadow_viewproj * pos;

// from homogeneous space to clip space
vec2 shadow_uv = proj_pos.xy / proj_pos.w;

// from clip space to uv space
shadow_uv = shadow_uv * 0.5 + vec2(0.5);

// get point depth (from -1 to 1)
float real_depth = proj_pos.z / proj_pos.w;

// normalize from [-1..+1] to [0..+1]
real_depth = real_depth * 0.5 + 0.5;

// read depth from depth buffer in [0..+1]
float shadow_depth = texture(shadowmap, shadow_uv).x;

// compute final shadow factor by comparing
float shadow_factor = 1.0;
if (shadow_depth < real_depth)
    shadow_factor = 0.0;
```

**Problems**:

- Resolution is limited on texture -> self-occlusion/… artifacts (-> floating shadows)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414005159308.png" alt="image-20220414005159308" style="zoom:67%;" /> 

- Depth precision is limited in texture

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414005215595.png" alt="image-20220414005215595" style="zoom:67%;" />  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414005238978.png" alt="image-20220414005238978" style="zoom: 67%;" />

### Basic Shading Solution

- Simple light + Ambient
- Blinn-Phong material
- Shadow map

-> Cheap, robust and easy modification

## Pre-computed Global Illumination

Trade time -> space 

### Representing Indirection Light

- Good compression rate
- Easy to do integration with material function

### Fourier Transform

> approaching any function with many orders of base functions

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220318170746669.png" style="zoom:67%;" />
$$
f(x) = \frac{A}{2} + \frac{2A\cos(t\omega)}{\pi} - \frac{2A\cos(3t\omega)}{3\pi} 
+ \frac{2A\cos(5t\omega)}{5\pi} - \frac{2A\cos(7t\omega)}{7\pi} + \cdots
$$

#### Convolution Theorem

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414005851427.png" alt="image-20220414005851427" style="zoom: 50%;" />

#### Spherical Harmonics

On a sphere (polar) coordinate: $x = \sin\theta\cos\phi$, $y = \sin\theta\cos\phi$, $z=\cos \theta$

The sperical harmonics: 
$$
Y_{lm} (\theta,\phi) = N_{lm} P_{lm} (\cos \theta)e^{Im\phi}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000606520.png" style="zoom: 50%;" />

Complex sphere integration can be approximated by quadratic polynomials:  
$$
\int_{\theta=0}^{\pi} \int_{\phi=0}^{2 \pi} L(\theta, \phi) Y_{l m}(\theta, \phi) \sin \theta d \theta d \phi \approx\left[\begin{array}{l}
x \\
y \\
z \\
1
\end{array}\right]^{T} M\left[\begin{array}{l}
x \\
y \\
z \\
1
\end{array}\right]
$$

> The shapes are similar to hydrogen electron orbitals
>
> ![image-20220411204351322](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411204351322.png)
>
> (Green - pos; Red - neg)

Spherical Harmonics, a mathematical system analogous to the Fourier transform but defined across the surface of a sphere. The SH functions in general are defined on imaginary numbers 

**Encoding**: (usually in game engine, 2-3 orders; when only requires to represent low-freq information only the first order is required)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411204516125.png" alt="image-20220411204516125" style="zoom: 67%;" />

Storage: Only requires <u>RGBA8 color</u> to store this information (i.e. = a RGB picture with alpha channel)

### SH Lightmap

#### Precomputed GI

- Parameterized all scene into huge 2D lightmap atlas
- Using offline lighting farm to calculate irradiance probes for all surface points
- Compress those irradiance probes into SH coefficients
- Store SH coefficients into 2D atlas lightmap textures  

#### UV Atlas (Baked)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411205201825.png" alt="image-20220411205201825" style="zoom:67%;" />

**Lightmap density**:

- Low-poly proxy geometry
- Fewer UV charts/islands
- Fewer lightmap texels are wasted

#### Lighting

- **Indirect lighting**
  - project lightmap from proxies to all LODs
  - apply mesh details
  - add short-range, high-freq lighting detail by HBAO
- **Direct + Indirect Lighting** 
  - compute direct lighting dynamically
- **Final Frame**
  - combined with materials

#### Pros & Cons

- **Pros**:
  - Very efficient on runtime
  - Bake a lot of fine detail of GI on env
- **Cons**:
  - Long and expensive precomputation (lightmap farm)
  - Only can handle static scene and static light
  - Storage cost on package and GPU (space)

### Light Probe: Probes in Game Space

> *Forza: Horizon* uses light probes

#### Generation

> traditionally by hand

- by terrain and road
- by voxel

=> final light probe cloud

#### Reflection Probe

High precision of sampling, low density

#### Pros & Cons

> Light probes + Reflection probes

- **Pros**:
  - Very efficient on runtime
  - Can be applied to both static and dynamic objects
  - Handle both diffuse and specular shading
- **Cons**: (not as good as shadow maps due to small amount of sampling)
  - A bunch of SH light probes need some pre-computation
  - Can not handle fine detail of GI. i.e., soft shadow on overlapped structures

## Physical-Based Materials (PBR)

### Microfact Theory

Key: the distribution of the microfacets’ normals (roughness)

- concentrated <==> glossy

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223113524613.png" style="zoom: 70%;" /> 

- spread <==> diffuse

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223113539391.png" style="zoom:67%;" /> 

### BRDF Model Based on Microfact

$$
f_r = k_df_{\text{Lambert}} + f_{\text{CookTorrance}}\\
\text{diffuse: } f_{\text{Lambert}} = \frac{c}{\pi}\, ; \quad
\text{specular: } f_{\text{CookTorrance}} = \frac{DFG}{4(\omega_o\cdot n)(\omega_i\cdot n)}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414113010543.png" alt="image-20220414113010543" style="zoom:50%;" />
$$
L_{o}\left(\vb{x}, \omega_{o}\right)=\int_{H^{2}}\left(k_{d} \frac{c}{\pi}+\frac{D F G}{4\left(\omega_{o} \cdot n\right)\left(\omega_{i} \cdot n\right)}\right) L_{i}\left(\vb{x}, \omega_{i}\right)\left(\omega_{i} \cdot n\right)\ \mathrm{d} \omega_{i}
$$
**DFG**: 3 terms of physical optics in GGX model

- **Normal Distribution Function (D)**: $\mathrm{NDF}_{\mathrm{GGX}} (n,h,\alpha) = \alpha^2 / (\pi((n\cdot h)^2(\alpha^2 - 1) + 1)^2)$ 

  also introduce <u>roughness</u> 

  Beckmann (red) / Phong (blue) / GGX (green)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414113307266.png" alt="image-20220414113307266" style="zoom:67%;" />

  ``` glsl
  // Normal Dist Func using GGX Dist
  float D_GGX(float NoH, float roughness) {
      float a2 = roughness * roughness;
      float f = (NoH * NoH) * (a2 - 1.0) + 1.0;
      return a2 / (PI * f * f);
  }
  ```

- **Geometric Attenuation Term (G)** (self-shadowing): 
  
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414113518197.png" alt="image-20220414113518197" style="zoom:80%;" />
  $$
  G_{\text{Smith} } (l,v) = G_{\mathrm{GGX}} (l)\cdot G_{\mathrm{GGX}}(v)\\
  G_{\mathrm{GGX}} (\nu) = \frac{n\cdot v}{(n\cdot v)(1-k) + k}\, ; \quad k = \frac{(\alpha+1)^2}{8}
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414114053462.png" alt="image-20220414114053462" style="zoom:80%;" />
  
  ``` glsl
  // Geometry term: Geometry masking/shadowing due to microfacets
  float GGX(float NdotV, float k) {
      return NdotV / (NdotV * (1.0 - k) + k);
  }
  
  float G_Smith(float NdotV, float NdotL, float roughness) {
      float k = pow(roughness + 1.0, 2.0) / 8.0;
      return GGX(NdotL, k) * GGX(NdotV, k);
  }
  ```
  
- **Fresnel Equation (F)**: 
  $$
  F_{\text{Schlick}}\left(h, v, F_{0}\right)=F_{0}+\left(1-F_{0}\right)(1-(v \cdot h))^{5}
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414114313563.png" alt="image-20220414114313563" style="zoom: 50%;" />
  
  ``` glsl
  // Fresnel term with scalar optimization
  float F_Schlick(float VoH, float f0) {
      float f = pow(1.0 - VoH, 5.0);
      return f0 + (1.0 - f0) * f;
  }
  ```

#### Physical Measured Material

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411211634086.png" alt="image-20220411211634086" style="zoom:67%;" />

MERL BRDF Database

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411211650273.png" alt="image-20220411211650273" style="zoom:80%;" />

#### Disney Principled BRDF

> for designers/artists (the real users are them)

Principles to follow when implementing model

- Intuitive rather than physical parameters should be used
- There should be as few parameters as possible
- Parameters should be zero to one over their plausible range
- Parameters should be allowed to be pushed beyond their plausible range where it makes sense
- All combinations of parameters should be as robust and plausible as possible

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220411212106711.png" alt="image-20220411212106711" style="zoom:80%;" />

### PBR SG and MR

#### PBR Specular Glossiness (SG)

> all properties are represented using graphs, complete model, obeying Disney’s principles (no need many parameters, just pictures are fine)

| Material [Final]                                             | Diffuse                                                      | Specular                                                     | Glossiness                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **SG**                                                       | RGB - sRGB                                                   | RGB - sRGB                                                   | Grayscale - Linear                                           |
| ![image-20220414222607963](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414222607963.png) | ![image-20220414222620297](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414222620297.png) | ![image-20220414222625954](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414222625954.png) | ![image-20220414222631988](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220414222631988.png) |

``` glsl
struct SpecularGlossiness {
    float3 specular;
    float3 diffuse;
    float3 normal;
    float glossiness;
}

SpecularGlossiness getPBRParameterSG() {
    SpecularGlossiness specular_glossiness;
    specular_glossiness.diffuse = sampleTexture(diffuse_texture, uv).rgb;
    specular_glossiness.specular = sampleTexture(specular_texture, uv).rgb;
    specular_glossiness.normal = sampleTexture(normal_texture, uv).rgb;
    specular_glossiness.glossiness = sampleTexture(glossiness_texture, uv).rgb;
    return specular_glossiness
}

// shader
float3 calculateBRDF(SpecularGlossiness specular_glossiness) {
    float3 half_vector = normalize(view_direction + light_direction);
    float N_dot_L = saturate(dot(specular_glossiness.normal, light_direction));
    float N_dot_V = abs(dot(specular_glossiness.normal, view_direction));
    float3 N_dot_H = saturate(dot(specular_glossiness.normal, half_vector));
    float3 V_dot_H = saturate(dot(view_direction, half_vector));
    
    // diffuse
    float3 diffuse = k_d * specular_glossiness.diffuse / PI;
    
    // specular
    float roughness = 1.0 - specular_glossiness.glossiness;
    float3 F0 = specular_glossiness.specular;
    
    float D = D_GGX(N_dot_H, roughness);        
    float3 F = F_Schlick(V_dot_H, F0);
    float G = G_Smith(N_dot_V, N_dot_L, roughness);
    float denominator = 4.0 * N_dot_V * N_dot_L + 0.001;
    
    float3 specular = (D * F * G) / denominator;
    
    // BRDF
    return diffuse + specular;
}

void PixelShaderSG() {
    SpecularGlossiness specular_glossiness = getPBRParameterSG();
    float3 brdf_reflection = calculateBRDF(specular_glossiness);
    return brdf_reflection * light_intensity * cos(light_incident_angle);
}
```

#### PBR Metallic Roughness (MR)

| Material [Final]                                             | Base Color                                                   | Roughness                                                    | Metallic                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **MR**                                                       | RGB - sRGB                                                   | Grayscale - Linear                                           | Grayscale - Linear                                           |
| ![image-20220415114540317](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415114540317.png) | ![image-20220415113414500](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415113414500.png) | ![image-20220415113430798](C:\Users\TR\AppData\Roaming\Typora\typora-user-images\image-20220415113430798.png) | ![](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415113439409.png) |

``` glsl
struct MetallicRoughness {
    float3 base_color;
    float3 normal;
    float roughness;
    float metallic;
}
```

#### Covert MR to SG

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415113605642.png" alt="image-20220415113605642" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415113737227.png" alt="image-20220415113737227" style="zoom: 50%;" />

``` glsl
SpecularGlossiness ConvertMetallicRoughnessToSpecularGlossiness (MetallicRoughness metallic_roughness){
    float3 base_color = metallic_roughness. base_color;
    float roughness = metallic_roughness. roughness;
    float metallic = metallic_roughness. metallic;
    
    float3 dielectricSpecularColor = float3(0.08f * dielectricSpecular);
    float3 specular = lerp(dielectricSpecularColor, base_color, metallic);
    float3 diffuse = base_color - base_color * metallic;
    
    SpecularGlossiness specular_glossiness;
    specular_glossiness. specular = specular;
    specular_glossiness. diffuse = diffuse;
    specular_glossiness. glossiness = 1.0f - roughness;
    
    return result;
}
```

#### PBR Pipeline MR vs SG

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415114253327.png" alt="image-20220415114253327" style="zoom: 50%;" />

- **MR**
  - Pros
    - Can be easier to author and less prone to errors caused by supplying incorrect dielectric F0 data
    - Uses less texture memory, as metallic and roughness are both grayscale maps
  - Cons
    - No control over F0 for dielectrics in map creation. However, most implementations have a specular control to override the base 4% value
    - Edge artifacts are more noticeable, especially at lower resolutions
- **SG**
  - Pros
    - Edge artifacts are less apparent
    - Control over dielectric F0 in the specular map
  - Cons
    - Because the specular map provides control over dielectric F0,  it is more susceptible to use of incorrect values. It is possible to break the law of conservation if handled incorrectly in the shader
    - Uses more texture memory with an additional RGB map  

## Image-Based Lighting (IBL)

### Basic Idea of IBL

- An image representing distant lighting from <u>all directions</u> 
- Shade a point under the lighting: Solving the <u>rendering equation</u> 
- Using <u>Monte Carlo integration</u> (large amount of <u>sampling</u> - slow!)

### Diffuse Irradiance Map

> Recall the rendering equation (contains the diffuse and specular terms)
> $$
> L_o(\vb{x}, \omega_o) = \int_{H^2} f_r(\vb{x}, \omega_o, \omega_i) L_i(\vb{x},\omega_i)\cos\theta_i\ \mathrm{d}\omega_i = L_d(\vb{x},\omega_o) + L_s (\vb{x},\omega_o) \\
> f_r = k_d f_{\text{Lambert}}+ f_{\text{CookTorrance}}
> $$

**Irradiance Map** 

> Use the pre-computed (pre-filtered/convoluted) map to store the lighting results

For the diffuse term (Use ${f_{\text{Lambert}} = \frac {c}{\pi}}$) 
$$
L_d({\vb{x},\omega_o}) = \int_{H^2} k_d f_{\text{Lambert}} L_i(\vb{x},\omega_i) \cos\theta_i \ \mathrm{d}\omega_i 
\approx
k_d^* c \frac{1}{\pi} \int_{H^2} L_i(\vb{x},\omega_i) \cos\theta_i\ \mathrm{d}\omega_i
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415173203891.png" alt="image-20220415173203891" style="zoom:80%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220415173218772.png" alt="image-20220415173218772" style="zoom:80%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416165125898.png" alt="image-20220416165125898" style="zoom: 40%;"  /> 

### Specular Approximation

The specular term: $\Rightarrow  (\alpha, F_0, \theta, \dots)$ 
$$
L_s (\vb{x},\omega_0) = 
\int_{H^2} f_{\text{CookTorrances}} L_i(\vb{x},\omega_i) \cos\theta_i\ \mathrm{d}\omega_i\\
f_{\text{CookTorrance}} = \frac{DFG}{4(\omega_o\cdot n)(\omega_i\cdot n)}
$$
Approximation by <u>Split Sum</u>:
$$
L_s (\vb{x},\omega_0) = 
\underbrace{\frac{\int_{H^2} f_{\text{CookTorrances}} L_i(\vb{x},\omega_i) \cos\theta_i\ \mathrm{d}\omega_i}{\int_{H^2} f_{\text{CookTorrances}}\cos\theta_i \ \mathrm{d}\omega_i}}_{\text{Lighting Term}} 
\cdot 
\underbrace{
\int_{H^2} f_{\text{CookTorrances}} \cos\theta_i\ \mathrm{d}\omega_i
}_{\text{BRDF Term}}
$$

- **Part 1**: <u>The lighting term</u> ($\alpha$ represents the <u>roughness</u>)
  $$
  \cdots \approx \frac{\sum ^N_k L(\omega_i^k) G(\omega_i^k)}{\sum^N_k G(\omega_i^k)}\, ,\quad \text{where } \alpha = G(\omega^k_i)  
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416175332075.png" alt="image-20220416175332075" style="zoom: 67%;" />

  -> Different roughness stored in a mipmap structure

- **Part 2**: <u>The BRDF term</u> (LUT - Lookup Table) 
  $$
  \begin{aligned}
  \cdots \approx &
  F_0 \int_{H^2} \underbrace{ \frac{f_{\text{CookTorrances}} }{F} \left(1- (1-\cos\theta_i)^5\right) \cos\theta_i \ \mathrm{d}\omega_i}_{(\alpha,\cos\theta_i)}  \\&+
  \int_{H^2} \frac{f_{\text{CookTorrances}} }{F}  (1-\cos\theta_i)^5 \cos\theta_i\ \mathrm{d}\omega_i\\
  &\approx F_0\cdot A+ B \approx F_0 \cdot \mathrm{LUT.r} + \mathrm{LUT.g}
  \end{aligned}
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416180116283.png" alt="image-20220416180116283" style="zoom:67%;" />

  -> Fresnel term becomes linear 

### Quick Shading with Pre-computation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416180408544.png" alt="image-20220416180408544" style="zoom: 50%;" />

> First time have specular in env highlight (but not the shiny ones)

## Classic Shadow Solution

### Cascade Shadow

#### Basic Ideas

- Partition the frustum into multiple frustums
- A shadow map is rendered for each sub frustum
- The pixel shader then samples from the map that most closely matches the required resolution  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416193716504.png" alt="image-20220416193716504" style="zoom:67%;" />

#### Steps

``` glsl
splitFrustumToSubfrusta();
calculateOrthoProjectionsForEachSubfrustum();
renderShadowMapForEachSubfrusum();
renderScene();

vs_main() {
    calculateWorldPosition()
    ....
}

ps_main() {
    transformWorldPositionsForEachProjections()
        sampleAllShadowMaps()
        compareDepthAndLightingPixel()
        ....
}
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416195153445.png" alt="image-20220416195153445" style="zoom:80%;" />

#### Blend between Cascade Layers

1. A visible seam can be seen where cascades overlap
2. Between cascade layers because the resolution does not match
3. The shader then linearly interpolates between the two values based on the pixel's location in the blend band  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416195354451.png" alt="image-20220416195354451" style="zoom: 80%;" />

#### Pros and Cons of Cascade Shadow

- **Pros**
  - best way to prevalent errors with shadowing: perspective aliasing
  - fast to generate depth map, 3x up when depth writing only
  - provide fairly good results
- **Cons**
  - Nearly impossible to generate high quality area shadows
  - No colored shadows. Translucent surfaces cast opaque shadows  

### Soft Shadows

#### Percentage Closer Filter (PCF)

> generates softer shadows

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416200153595.png" alt="image-20220416200153595" style="zoom:80%;" />

- **Target problem**: The shadows that result from shadow mapping aliasing is serious  
- **Basic idea**:
  - Sample from the shadow map around the current pixel and compare its depth to all the samples
  - By averaging out the results we get a smoother line between light and shadow  

#### Percentage Closer Soft Shadow (PCSS)

- **Target problem**: Suffers from aliasing and under sampling artifacts  
- **Basic Idea**:
  - Search the shadow map and average the depths that are closer to the light source
  - Using a parallel planes approximation  

$$
w_{\text {Penumbra }}=\frac{\left(d_{\text {Receiver }}-d_{\text {Blecker }}\right) \cdot w_{\text {Light }}}{d_{\text {Blocker }}}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416200422651.png" alt="image-20220416200422651" style="zoom: 25%;" />

#### Variance Soft Shadow Map

- **Target problem**: Rendering plausible soft shadow in real-time

- **Basic idea**:

  ​	Based on Chebyshev‘s inequality, using the average and variance of depth, we can approximate the percentage of depth distribution directly instead of comparing a single depth to a particular region (PCSS)  

$$
\begin{aligned}
\mu &= E(x)\\
\sigma^2 &= E(x^2) - E(x)^2
\end{aligned}
$$

![image-20220416201755939](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416201755939.png)

### Summary of Popular AAA Rendering

- Lightmap + Light Probe
- PBR + IBL
- Cascade shadow + VSSM

## Moving Wave of High Quality

### Quick evolving of GPU

- More flexible new shader model
  - Compute shader
  - Mesh shader
  - Ray-tracing shader
- High performance parallel architecture
  - Warp or wave architecture  
- Fully opened graphics APIs: DirectX 12 & Vulkan

### Hardware/Algorithm Updates

- Real-time ray-tracing on GPU
- Real-time global illumination (SS, SDF, SVOGI/VXGI, RSM/RTX)
- More complex material model: BSDF (strand-based hair), BSSRDF
- Virtual shadow maps (UE5) -> solve the waste of Cascade shadow

## Shader Management

> Ocean of shaders

- Artists create infinite more shaders  

### Uber shader and variants

A combination of shader for all possible light types, render passes and material types  

- Shared many state and codes
- Compile to many variant short shaders by pre-defined macro  

### Cross Platform Shader Compile

Shader Cross-Compiler Flow  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220416205403160.png" alt="image-20220416205403160" style="zoom: 33%;" />
