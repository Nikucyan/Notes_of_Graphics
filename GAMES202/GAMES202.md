# GAMES202 - Real-Time High Quality Rendering

Lecturer: [Lingqi Yan](www.cs.ucsb.edu/~lingqi/) 



---



# Lecture 1 Introduction and Overview

## Topics

**Real-Time High Quality Rendering**

- **Real-Time**
  - Speed: more than <u>30 fps</u> (VR/AR: 90 fps)
  - Interactivity: each frame generated <u>on the fly</u>
- **High Quality**
  - Realism: advanced approaches to make rendering more realistic
  - Dependability: all-time <u>correctness (exact or approx)</u>; no tolerance to (uncontrollable) failures
- **Rendering**

**Main Topics**

Highest Level: 4 parts on real-time rendering

<u>Shadows</u> (and env) / <u>Global Illum.</u> (Scene/image space, precomputed) / <u>Physically-based Shading</u> / <u>Real-time Ray Tracing</u>



# Lecture 2 Recap of CG Basics

## Basic GPU Hardware Pipeline

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726163726899.png" style="zoom:67%;" />

## OpenGL

OpenGL is a set of APIs that call the GPU pipeline from CPU (language does not matter, cross platform, alternatives (DirectX, Vulkan, etc.))

### **Analogy**: oil painting  

- **Place objects/models**: Model specification / transformation

  User specifies an object’s vertices, normals, texture coords and send them to GPU as a  Vertex buffer object (VBO) (similar to `.obj` files)

  OpenGL functions to obtain matrices: `glTranslate`, `glMultMatrix`, etc. 

- **Set position of an easel**: View transformation; Create / use a <u>framebuffer</u>

  Set cam by simply calling, e.g., `gluPerspective  ` 

- **Attach a canvas to the easel**: One rendering <u>pass</u> in OpenGL

  A framebuffer is specified to use and specify one or more textures as output (shading / depth / …) => Multiple render target 

- **Paint to the canvas**: (how to perform shading), when vertex / fragment shaders will be used

  - For each <u>vertex</u> in parallel: OpenGL calls user-spec. vertex shader: Transform vertex (ModelView, Projection), other ops
  - For each primitive, OpenGL <u>rasterizes</u>: Generates a <u>fragment</u> for each pixel the fragment covers  
  - For each <u>fragment</u> in parallel: OpenGL calls user-specified fragment shader (Shading and lighting calculations); handles z-buffer depth test unless overwritten  

- (Attach other canvases to the easel and continue painting)

- (Use previous paintings for reference)  

### Summary

In each pass:

- Specify objects, camera, MVP, etc.
- Specify framebuffer and input/output textures
- Specify vertex / fragment shaders
- (When you have everything specified on the GPU) Render!  

Left: <u>Multiple passes</u> (Use your own previous paintings for reference)  <= shadow mapping

## OpenGL Shading Language (GLSL)

### Shading Languages  

- Vertex / Fragment shading described by small program
- Written in language similar to C but with restrictions  

Different languages => need to compile

- In ancient times: assembly on GPUs (extremely hard)
- Stanford Real-Time Shading Language, work at SGI
- Still long ago: Cg from NVIDIA
- **HLSL** in <u>DirectX</u> (vertex + pixel)
- **GLSL** in <u>OpenGL</u> (vertex + fragment)  

### Shader Setup

- Initializing 
  - Create shader (vertex and fragment)
  - Compile shader
  - Attach shader to program
  - Link program
  - Use program
- Shader source is sequence of strings
- Similar steps to compile a normal program

### Linking Shader Program

``` glsl
GLuint initshaders (GLenum type, const char *filename) {
	// Using GLSL shaders, OpenGL book, page 679
	GLuint shader = glCreateShader(type) ;
	GLint compiled ;
	string str = textFileRead (filename) ;
	GLchar * cstr = new GLchar[str.size()+1] ;
	const GLchar * cstr2 = cstr ; // Weirdness to get a const char
	strcpy(cstr,str.c_str()) ;
	glShaderSource (shader, 1, &cstr2, NULL) ;
	glCompileShader (shader) ;
	glGetShaderiv (shader, GL_COMPILE_STATUS, &compiled) ;
	if (!compiled) {
		shadererrors (shader) ;
		throw 3 ;
	}
	return shader ;
}
```

### Phong Shader in Assignment 0

- **Vertex Shader**: (inloop) 

  ``` glsl
  attribute vec3 aVertexPosition;
  attribute vec3 aNormalPosition;
  attribute vec2 aTextureCoord;
  
  uniform mat4 uModelViewMatrix; // uniform for global var
  uniform mat4 uProjectionMatrix;
  
  varying highp vec2 vTextureCoord;
  varying highp vec3 vFragPos;
  varying highp vec3 vNormal;
  
  void main(void) {
      
      vFragPos = aVertexPosition;
      vNormal = aNormalPosition;
      
      gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);
      // vec4 can be used as vec4([vec3], a);
      
      vTextureCoord = aTextureCoord;
  }
  ```

- **Fragment Shader**: (can spec write to some texture)

  ``` glsl
  // ...
  
  uniform sampler2D uSampler;	// def texture 
  uniform vec3 uKd;
  uniform vec3 uKs;
  uniform vec3 uLightPos;
  uniform vec3 uCameraPos;
  uniform float uLightIntensity;
  uniform int uTextureSample;
  
  varying highp vec2 vTextureCoord;
  varying highp vec3 vFragPos;
  varying highp vec3 vNormal;
  
  void main(void) {
      vec3 color;
      if (uTextureSample == 1) {
          color = pow(texture2D(uSampler, vTextureCoord).rgb, vec3(2.2));	// this pow 2.2 for gamma correction
      } else {
          color = uKd;
      }        
      vec3 ambient = 0.05 * color;
      
      vec3 lightDir = normalize (uLightPos - vFragPos);
      vec3 normal = normalize(vNormal);
      float diff = max (dot(lightDir, normal), 0.0);
      float light_atten_coff = uLightIntensity / length (uLightPos - vFragPos);
      vec3 diffuse = diff * light_atten_coff * color;
      
      vec3 viewDir = normalize(uCameraPos - vFragPos);
      float spec = 0.0;
      vec3 reflectDir = reflect (-lightDir, normal);
      spec = pow (max(dot(viewDir, reflectDir), 0.0), 35.0);
      vec3 specular = uKs * light_atten_coff * spec;
      
      gl_FragColor = vec4 (pow((ambient + diffuse + specular), vec3(1.0/2.2)), 1.0);
      
  }
  ```

### Debugging Shaders

- Years ago: 
  - NVIDIA Nsight with VS (Needed multiple GPUS, had to run in software sim mode in HLSL)
- Now: 
  - Nsight Graphics (NV GPUs only)
  - RenderDoc (no lim on GPUs)
- Debug without debugging tools:
  - Print it out (<u>show values (depth / …) as colors</u>, cannot use `cout` / `printf`)   

## The Rendering Equation

Describing light transport (from [GAMES101-Lec.15](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#the-rendering-equation))
$$
\underbrace{L_{o}\left(\mathrm{p}, \omega_{o}\right)}_{\text{Outgoing Radiance}} =
\underbrace{L_{e}\left(\mathrm{p}, \omega_{o}\right)}_{\text{Emission}} +
\int_{H^2} 
\underbrace{L_{i}\left(\mathrm{p}, \omega_{i}\right)}_{\begin{aligned}&\text{Incident Light}\\&(\text{from source}) \end{aligned}}
\underbrace{f_{r}\left(\mathrm{p}, \omega_{i}\rightarrow \omega_{o}\right)}_{\text{BRDF}}
\cos\theta_i
\ \mathrm{d} \omega_{i}
$$

- In real-time rendering (RTR): 

  - **Visibility** is often explicitly considered 
  - **BRDF** is often considered with the **cos** term together

  $$
  \underbrace{L_{o}\left(\mathrm{p}, \omega_{o}\right)}_{\text{Outgoing Light}} =
  \underbrace{L_{e}\left(\mathrm{p}, \omega_{o}\right)}_{\text{Emission}} +
  \int_{\Omega^+}  
  \underbrace{L_{i}\left(\mathrm{p}, \omega_{i}\right)}_{\begin{aligned}&\text{Incident Light}\\&(\text{from source}) \end{aligned}}
  \underbrace{f_{r}\left(\mathrm{p}, \omega_{i}\rightarrow \omega_{o}\right)\cos\theta_i}_{\begin{aligned}&\text{(Cosine-Weighted)}\\ &\qquad \text{BRDF}\end{aligned}}\
  \underbrace{V(\mathrm{p},\omega_i)}_{\text{Visibility}}
  \ \mathrm{d} \omega_{i}
  $$

### Environment Lighting

Representing incident lighting from all directions

- Usually represented as a cube map or a sphere map (texture)  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305172429501.png" alt="image-20220305172429501" style="zoom: 43%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305172500908.png" alt="image-20220305172500908" style="zoom: 50%;" /> 

Difficulty in RTR: 

- for every fragment need to solve the integrals in the rendering equation;

- Multiple-time bounces global illumination (dir / indir) 

  usually dir illum + 1 time indir glb illum



# Lecture 3-4 Real-time Soft Shadows

## Recap: Shadow Mapping

- A <u>2-pass</u> Algorithm
  - The light pass generates the SM
  - The cam pass uses the SM (recall rendering equation)
- An <u>Image-space</u> Algorithm 
  - Pro: no knowledge of scene’s geometry is required
  - Con: causing self occlusion and aliasing issues
- Well known shadow rendering technique (even for early offline renderings)

### The Two Passes

- Pass 1: Rendering from light

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173441462.png" alt="image-20211217173441462" style="zoom: 50%;" /> 

- Pass 2: Project to light for shadows

  Project visible points in eye view back to light source
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173727667.png" alt="image-20211217173727667" style="zoom: 50%;" /> 

  - (Reprojected) depths match for light and eye: <u>VISIBLE</u> 
  - (Reprojected) depths from light, eye not the same: <u>BLOCKED</u>

### Visualizing Shadow Mapping

- The depth buffer from the light’s point-of-view

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305210047212.png" alt="image-20220305210047212" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305210057277.png" alt="image-20220305210057277" style="zoom:67%;" /> 

- Projecting the depth map onto the eye’s view  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305210233367.png" alt="image-20220305210233367" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305210241931.png" alt="image-20220305210241931" style="zoom:67%;" />  

### Issue in Shadow Mapping

- **Self Occlusion** (strong moire patterns) 

  In one fragment, the light source pass and the eye pass have different depths with smal deviation in the projection on the pixel

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305211024213.png" alt="image-20220305211024213" style="zoom: 37%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305211058507.png" alt="image-20220305211058507" style="zoom: 45%;" />

  The most severe case: grazing angle (longest shadow)

- **Adding a (variable) <u>bias</u>** to reduce self occlusion (but introducing **detached shadow** issue)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305212449533.png" alt="image-20220305212449533" style="zoom:37%;" /> 
  
- **Second-depth Shadow Mapping**: Not only store the smallest depth, but also the second smallest one

  Use the midpoint between first and second depths, but requires objects to be watertight and costs too high overhead

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220305213422169.png" alt="image-20220305213422169" style="zoom:40%;" /> 

  => RTR doesn’t trust in complexity (only real speed matters)

- **Aliasing**

  Shadow map has <u>resolution</u> 

  => solutions: dynamic resolution, …

## The Math Behind Shadow Mapping

### Inequalities in Calculus

For $f(x)$ and $g(x)$ that can be integraled on $[a,b]$ 

- Schwarz inequality
  $$
  \left[\int_{a}^{b} f(x) g(x)\ \mathrm{d} x\right]^{2} \leqslant \int_{a}^{b} f^{2}(x) \ \mathrm{d} x \cdot \int_{a}^{b} g^{2}(x)\ \mathrm{d} x
  $$

- Minkowski inequality
  $$
  \left\{\int_{a}^{b}[f(x)+g(x)]^{2} \mathrm{~d} x\right\}^{\frac{1}{2}} \leqslant\left\{\int_{a}^{b} f^{2}(x)\ \mathrm{d} x\right\}^{\frac{1}{2}}+\left\{\int_{a}^{b} g^{2}(x)\ \mathrm{d} x\right\}^{\frac{1}{2}}
  $$

### Approximation in RTR

In RTR, care more about “approximately equal”

An important approx:
$$
\int_{\Omega} f(x) g(x)\ \mathrm{d} x \approx \frac{\int_{\Omega} f(x)\ \mathrm{d} x}{\int_{\Omega}\ \mathrm{d} x} \cdot \int_{\Omega} g(x)\ \mathrm{d} x
$$

($1/\int _\Omega \ \mathrm{d}x$ is the const for normalizing) 

The conditions that make this approx more accurate: (one is enough)

- The support of $g(x)$ is small enough
- $g(x)$ is smooth enough

Recall the **rendering equation** (the visibility part)
$$
\begin{aligned}
L_{o}\left(\mathrm{p}, \omega_{o}\right)&=
L_{e}\left(\mathrm{p}, \omega_{o}\right)+
\int_{\Omega^+}  
L_{i}\left(\mathrm{p}, \omega_{i}\right)\
f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\
\cos\theta_i\
V(\mathrm{p},\omega_i)
\ \mathrm{d} \omega_{i}\\
&\approx 
\frac{\int_{\Omega^+} V(\mathrm{p},\omega_i)\ \mathrm{d}\omega_i}{\int_{\Omega^+} \mathrm{d}\omega_i }
\cdot 
\int_{\Omega^+}  
L_{i}\left(\mathrm{p}, \omega_{i}\right)\
f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\
\cos\theta_i\ \mathrm{d}\omega_i
\end{aligned}
$$
The conditions make more accurate

- Small support (<u>point / directional lighting</u>) => hard shadow
- Smooth integrand (<u>diffuse bsdf / const radiance area lighting</u>)

=> ambient occlusion

## Percentage Closer Soft Shadows (PCSS)

Hard shadows and soft shadows:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306111457560.png" alt="image-20220306111457560" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306111530320.png" alt="image-20220306111530320" style="zoom:67%;" />

### Percentage Closer Filtering (PCF)

Provides **anti-aliasing** at shadows’ edges: Filtering the results of shadow comparisons (not for soft shadows)

Why not filtering the shadow map:

- Texture filtering just averages <u>color components</u>, i.e. get <u>blurred</u> shadow map first
- Averaging <u>depth values</u>, then <u>comparing</u>, still get a <u>binary visiblitiy</u> 

#### The Solution

- Perform multiple (e.g., 7x7) depth comparisons for each fragment

- Then average results of comparisons

- e.g., for pt P on the floor

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306155249215.png" alt="image-20220306155249215" style="zoom:53%;" /> 

  - compare its depth with all pixels in the box, e.g., 3x3

  - get the compared results, e.g., 

    $\begin{matrix}1 & 0 & 1\\ 1 & 0 & 1 \\ 1& 1& 0 \end{matrix}$ 

  - take average to get visibility, e.g., 0.667

#### Effects

- The filter size: smaller -> sharper; bigger -> softer
- Use large PCF can achieve soft shadow effects
- Key thoughts: from hard to soft shadows; the correct size to filter; uniform or not

### Percentage Closer Soft Shadows (PCSS)

#### Key Idea

- Filter size <-> blocker distance
- **Relative average projected blocker depth**

$$
w_{\text {Penumbra}}=\left(d_{\text {Receiver}}-d_{\text {Blocker}}\right) \cdot w_{\text {Light}} / d_{\text {Blocker}}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306160928613.png" alt="image-20220306160928613" style="zoom:40%;" />

#### Algorithm

- Step 1: **Blocker search**: getting the average blocker depth in a <u>certain region</u>
- Step 2: **Penumbra estimation**: use the aveerage blocker depth to determine filter size
- Step 3: **Percentage Closer Filtering** (PCF)

=> can be very slow (both step 1 and 3 need to compare <u>all texels</u> in the spec region)

#### Region for Blocker Search

The region to perform blocker search: can be set const (e.g., 5x5) but can be better with heuristics

- Depend on the light size
- Depend on the receiver’s distance from the light

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306162254493.png" alt="image-20220306162254493" style="zoom: 33%;" />

### A Deeper Look at PCF

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220306231223014.png" alt="image-20220306231223014" style="zoom: 40%;" />

- **Filter / Convolution**: (the weighted average on the neighbor region of point $p$ (as $q$))
  $$
  [w * f] (p) = \sum_{q\in \mathcal{N}(p)} w (p,q) f(q)
  $$

- **PCSS**: (the last term for the comparison results of the occlusion: (sign/step func) true -> visibility = 1; false (no occlusion) -> 0)
  $$
  V(x) = \sum_{q\in \mathcal{N}(p) } w(p,q) \cdot \chi^+ [D_{\text{SM}}(q) - D_{\text{scene}}(x)]
  $$
  Therefore, PCF is not filtering the shadow map then compare
  $$
  V(x) \neq \chi^+ \{[w*D_{\text{SM}}](q) - D_{\text{scene}}(x) \}
  $$
  And PCF is not filtering the result img with binary visibilities
  $$
  V(x) \neq \sum_{q\in \mathcal{N}(q)}w(p,q)V(q)
  $$

## Variance Soft Shadow Mapping (VSSM)

Fast block search (step 1) and filtering (step 3) => solve the problems of PCSS

> From the algorithm of PCF:
>
> - The percentage of texels that are in front of the shading point, i.e.,
> - How many texels are closer than t in the search area, i.e.,
> - how many students did better than you in an exam  
>   - Using a <u>histogram</u> -> <u>accurate</u> answer
>   - Using a <u>normal dist</u> -> <u>approx</u> answer

### Key Idea

- Quickly compute the **mean** and **variance** of depths in an area

  - **Mean**: hardware MIPMAPing / Summed Area Tables (SAT) 

  - **Variance**: $\text{Var} (X) = \text{E}(X^2) - \text{E}^2(X)$ (variance connected with expects)

    need the mean of (depth^2^) => generate a “<u>square-depth map</u>”along the shadow map

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220307192723199.png" alt="image-20220307192723199" style="zoom: 43%;" />

=> error function `erf()` -> the numerical sol for this integral

### Approximation

**Chebychev’s** inequality (one-tailed version, for $t > \mu$) (where $\mu $ - mean; $\sigma^2$ - variance) => use as an approximation
$$
P(x>t) \le \frac{\sigma^2}{\sigma^2+(t-\mu)^2}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220309004038622.png" alt="image-20220309004038622" style="zoom: 33%;" />

The condition is $t$ should be on the **right** side of the mean value, or the result will be too inaccurate

### Process

- Performance
  - Shadow map generation
    - “Square depth map”: parallel, along with shadow map, #pixels
    - MIPMAP / SAT / …
  - Run time (no samples / loops needed)
    - Mean of depth in a range $\mathcal O (1)$
    - Mean of depth square in a range: $\mathcal{O}(1)$
    - Chebychev: $\mathcal{O}(1)$ 
- Step 3 (filtering) solved perfectly

> but we want step 1 and 3 both to be solved better, now step 3 is better but step 1 still has problem  

### Problem in Step 1

**blocker search** within an area => loop / sample, inefficient

We want the average depth of blockers (not the avg depth $z_{\text{avg}}$)

**Key idea**: 

- blocker $(z<t)$, avg. $z_{\text{occ}}$ (the depth of blocker < shading point)
- non-blocker $(z>t)$, avg. $z_{\text{unocc}}$ 


$$
\frac{N_1}{N} z_{\text{unocc}} + \frac{N_2}{N} z_{\text{occ}} = z_{\text{avg}} 
$$

=> Approximation: $N_1 / N = P(x>t)$, **Chebychev**; $N_2 / N = 1-P(x-t)$ 

but we don’t know $z_{\text{unocc}}$ => Approximation: $z_{\text{unocc}} = t$ (i.e. shadow receiver is a plane)

> But VSSM not as popular as PCSS this time <= PCSS could be noisy but the denoise techniques are now mature enough

## MIPMAP and Summed-Area Variance Shadow Maps

In VSSM: in order to accelerate, need to quickly grab $\mu$ and $\sigma$ from an arbitrary range (rectangular)
$$
P(x\ge t)\le p_{\max} (t) \equiv \frac{\sigma^2}{\sigma^2 + (t-\mu)^2} 
$$
For the average $\mu$, this is rectangular range query -> can be handled by both MIPMAP and Summed-Area Table (SAT)

### MIPMAP for Range Query

> MIPMAP (in GAMES101): Allowing **fast, approx., square** <u>range queries</u> (still approximate even with trilinear interpolation (anisotropic query can help))

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312103852811.png" alt="image-20220312103852811" style="zoom: 50%;" />

### SAT for Range Query

> Summed-Area Table (Data Structure)

Classic data structure and algorithm (<u>prefix sum</u>)

- In 1D: every step sum everything on the leftside up (including itself): convert the addition of several numbers to the substraction of 2 numbers

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312103552447.png" alt="image-20220312103552447" style="zoom: 50%;" /> 
  
- In 2D:
  
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312104307394.png" alt="image-20220312104307394" style="zoom:80%;" /> 
  
  Accurate, but need $\mathcal{O}(n)$ time and storage to build => storage might not be an issue, want to speed up SAT

## Moment Shadow Mapping  

### Revisit: VSSM

<u>Normal distribution</u> (including the approx of Chebychev) is not always good enough to approx the distribution of fragments’ distances

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312161006473.png" alt="image-20220312161006473" style="zoom:80%;" />

For complex occlusion the approx of normal dist is OK, but for very simple scenarios not good

**Issues** if inaccurate: <u>Overly dark</u> (may be acceptable); <u>Overly bright: Light leaking</u>

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312161236554.png" alt="image-20220312161236554" style="zoom: 33%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312161404188.png" alt="image-20220312161404188" style="zoom: 41%;" /> 

**Limitations**: <u>Light leaking</u> (bleeding), <u>non-planarity artifact</u> 

Chebychev: only valid when $t>z_{\text{avg}}$ 

### Moment Shadow Mapping

**Goal**: Represent a <u>distribution</u> more accurately (but still not too costly to store) (just improve the distribution from Chebychev)

**Idea**: Use <u>higher order moments</u> to represent the dist

Definition of Moments here: Use the simplest one: $x,\ x^2,\ x^3,\ x^4$ (VSSM is essentially using the <u>first two orders</u> of moments => In MSM, use higher order)

The first $ m$ orders of moments can represent a function with $m/2$ steps. Usually, 4 is good enough to approx the actual CDF of depth test

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312162000704.png" alt="image-20220312162000704" style="zoom: 33%;" />

- Pros: Very nice results
- Cons: Costly storage (may be fine); Costly performance (in the reconstruction)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312170905389.png" alt="image-20220312170905389" style="zoom: 33%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220312170856608.png" alt="image-20220312170856608" style="zoom: 33%;" />

## Distance Field Soft Shadows

> Fast but high costs of storage

### Distance Functions

> From [GAMES101 - Lec 10](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#representing-ways) 

At any point, giving the **minimum distance** (could be <u>signed</u> distance) to the <u>closest</u> location on an object (a scalar field)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313101813024.png" alt="image-20220313101813024" style="zoom: 33%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313101822928.png" alt="image-20220313101822928" style="zoom: 33%;" />

An example: blending (linear interp.) a moving boundary ($\mathrm{lerp}(A,B,0.5) $ -> grey, not wanted => use SDF -> lerp will obtain a value at the middle)  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313101930309.png" alt="image-20220313101930309" style="zoom: 67%;" />

Can blend any two distance function $d_1$, $d_2$ (optimal transport)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803161822644.png" style="zoom:50%;" />

### The Usage of DF

- **Ray marching** (sphere tracing) to perform <u>ray-SDF intersection</u>

  The value of SDF == a “safe” distance around

  Each time at p, just travel SDF(p) distance

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313102743994.png" alt="image-20220313102743994" style="zoom:80%;" /> 

  Good property for multiple objects: just find the smallest SDF at anypoint (= the smallest distance to the closest object in the whole scene) (but not good for deformations)

- **Generate soft shadows**: determine the (approx) <u>percentage of occlusion</u>

  The value of SDF -> a “safe” angle seen from the eye

  Smaller “safe” angle <-> less visibility

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313103434535.png" alt="image-20220313103434535" style="zoom:80%;" /> 

### Distance Field Soft Shadows

During ray matching: calculate the <u>“safe” angle</u> from the eye at every step; keep <u>minimum</u>; how to compute the angle

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313104215952.png" alt="image-20220313104215952" style="zoom:80%;" />
$$
\arcsin\frac{\mathrm{SDF(p)}}{|p-o|}\;\; \text{ or }\;\; \min\left\{\frac{k\cdot \mathrm{SDF}(p)}{|p-o|},1.0 \right\}
$$

> Can also use sigmoid but more complex

=> Larger $k$ <-> earlier cutoff of penumbra <-> harder (k to control the hardness of shadows)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313104542168.png" alt="image-20220313104542168" style="zoom: 67%;" />

**Visualization**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313104720600.png" alt="image-20220313104720600" style="zoom:33%;" />

**Pros and Cons**

- **Pros**: Fast; High quality
- **Cons**: Need precomputation; Need heavy storage; Artifact; Not good for texture mapping

**Another Application**: antialiased / infinite resolution characters in RTR



# Lecture 5-7 Real-time Environment Mapping

## Shading from Environment Lighting

### Recap of Environment Lighting

> GAMES101

An image representing distant lighting from all dir (infinitely far); spherical / cube map

> Informally named **Image-Based Lighting (IBL)**  

To shade a point (without shadows): solving the **rendering equation** (lighting + BRDF (with cos), can neglect visibility) 

General solution: [**Monte Carlo** integration](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#monte-carlo-integration) (numerical and requires large amount of samples => slow (generally sampling not preferred in shaders)) 

For BRDF: If <u>glossy</u> - <u>small support;</u> if <u>diffuse</u> - <u>smooth</u>

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313131008616.png" alt="image-20220313131008616" style="zoom: 33%;" />

### The Classic Approximation

Recall: Note the slight edit on $\Omega_G$ here
$$
\int_{\Omega} f(x) g(x)\ \mathrm{d} x \approx \frac{\int_{\Omega} f(x)\ \mathrm{d} x}{\int_{\Omega}\ \mathrm{d} x} \cdot \int_{\Omega} g(x)\ \mathrm{d} x
$$
Conditions for acceptable accuracy: $g(x)$ small or smooth (suitable for BRDF) => split the term

### The Split Sum Approximation

#### 1st Stage

BRDF satisfies the accuracy condition in any case. we can safely take the **lighting** term out

> In shadows, we take the visibility term out (keep lighting and BRDF terms)

$$
L_o(p,\omega_o) \approx \underbrace{\frac{\int_{\Omega_{f_r}} L_i(p,\omega_i)\ \mathrm{d} \omega_i}{\int_{\Omega_{f_r}} \mathrm{d} \omega_i} }_{\text{First stage}}
\cdot \underbrace{\int_{\Omega^+} f_r(p,\omega_i,\omega_o)\cos\theta_i\ \mathrm{d} \omega_i}_{\text{Second Stage}}
$$

=> **Prefiltering** of the environment lighting (take average)

- Pre-generating a set of differently filtered environment lighting
- Filter size in-between can be approximated via trilinear interp. (~ MIPMAP)

Then query the pre-filtered environment lighting at the <u>$r$ (mirror reflected) direction</u> (the filtered result to be used in query, take the mirror reflected direction point to represent the filtered region)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313162758392.png" alt="image-20220313162758392" style="zoom: 33%;" />

#### 2nd Stage

The first term now doesn’t have sampling, need to figure out the second term (still has an integral)

**Idea**: Precompute the value for all possible combinations of variables <u>roughness</u> (NDF), <u>color</u> (Fresnel term), etc (But need a huge table with extremely high dimensions (5D table))

>  Recall: [Microfacet BRDF](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#microfacet-brdf) ($\vb{F}$ - Fresnel term; $\vb{G}$ - shadow masking term; $\vb{D}$ - distribution of normals)
>  $$
>  f(\vb{i,o}) = \frac{\vb{F(i,h)G(i,o,h)D(h)}}{4\vb{(n,i)(n,o)}}
>  $$
>  The [Fresnel term approximation](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#fresnel-reflection--term): Schlick’s approximation: 
>
>  ​	$R(\theta) = R_0+(1-R_0)(1-\cos\theta)^5$; $R_0 = ((n_1-n_2)/(n_1+n_2))^2$  (Def.ed reflectance rate $R_0$ and the upward trend)
>
>  The [NDF term](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#detailed-appearance): e.g. Beckmann dist (Similar to Gaussian, var $\alpha$ - roughness and $\theta_h$ - angle between half vector and normal)
>  $$
>  \vb{D(h)} = \frac{e^{-\tan^2 \theta_h/a^2}}{\pi \alpha^2 \cos^4\theta_h}
>  $$

**Idea**: Try to split the variables again; the Schlick approx Fresnel term is much simpler, just “base color” $R_0$ and half angle $\theta$

Taking the approx into 2nd term: (extract the “base color” ($R_0$ is a const))
$$
%\begin{aligned}
\int_{\Omega^{+}} f_{r}\left(p, \omega_{i}, \omega_{o}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \approx R_{0} \int_{\Omega^{+}} \frac{f_{r}}{F}\left(1-\left(1-\cos \theta_{i}\right)^{5}\right) \cos \theta_{i} \mathrm{~d} \omega_{i}+%\\
 \int_{\Omega^{+}} \frac{f_{r}}{F}\left(1-\cos \theta_{i}\right)^{5} \cos \theta_{i} \mathrm{~d} \omega_{i}
%\end{aligned}
$$
Each integral produces one value for each (roughnessm incident angle) pair => each integral result in a 2D table (texture) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313171426345.png" alt="image-20220313171426345" style="zoom: 50%;" />

#### Properties

- Completely avoided sampling
- Very fast and almost identical results

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220313171550921.png" alt="image-20220313171550921" style="zoom: 67%;" />

#### In the Industry

Integral -> Sum 
$$
\frac{1}{N} \sum_{k=1}^{N} \frac{L_{i}\left(\mathbf{l}_{k}\right) f\left(\mathbf{l}_{k}, \mathbf{v}\right) \cos \theta_{\mathbf{l}_{k}}}{p\left(\mathbf{l}_{k}, \mathbf{v}\right)} \approx\left(\frac{1}{N} \sum_{k=1}^{N} L_{i}\left(\mathbf{l}_{k}\right)\right)\left(\frac{1}{N} \sum_{k=1}^{N} \frac{f\left(\mathbf{l}_{k}, \mathbf{v}\right) \cos \theta_{\mathbf{l}_{k}}}{p\left(\mathbf{l}_{k}, \mathbf{v}\right)}\right)
$$

> Unreal Engine’s split sum method in PBR 

## Shadow from Environment Lighting

- In general, it’s very hard for RTR
- Different perspectives of view: 
  - As a <u>many-light problem</u>: cost of SM is linearly to #light
  - As a <u>sampling problem</u>: visibility term $V$ can be arbitrarily complex and $V$ cannot be easily separated from the env
- Industrial solution
  - Generate one (or a little bit more) shadows from the brightest light source

- Related research
  - Imperfect shadow maps
  - Light cuts (reflecting objects as light sources - sorting): offline many-light problem
  - RTRT (might be the ultimate solution)
  - <u>Precomputed radiance transfer</u> (accurate)


 ### Background Knowledge

#### Frequency and Filtering

> From [GAMES101](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#frequency-domain)

##### Fourier Transform (Expansion)

Every term in the expansion has different frequency (increasing). The expansion represents a function as a weighted sum of sines and cosines (all this functions are called <u>basis functions</u>) 
$$
f(x) = \frac{A}{2} + \frac{2A\cos (t\omega)}{ \pi} -\frac{2A\cos(3t\omega)}{3\pi} + \frac{2A\cos(5t\omega)}{5\pi} - \frac{2A\cos(7t\omega)}{7\pi} + \cdots
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220318170746669.png" alt="image-20220318170746669" style="zoom:67%;" />

##### Visualizing Image Frequency Content

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220318171405110.png" alt="image-20220318171405110" style="zoom: 33%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220318171934039.png" alt="image-20220318171934039" style="zoom: 33%;" />

Close to the center - low freq; far from the center - high freq

##### Filtering 

Filtering: Getting rid of certain frequency contents

Low-pass filter makes the image blurred

**Convolution theorem**: convolution in the spatial domain is multiplication in the frequency domain

=> Any **product integral** can be considered as **filtering**: $\int_{\Omega} f(x)g(x)\ \mathrm{d}x$ 

​	Low freq == smooth function / slow changes / etc

​	The freq of the integral is the <u>lowest of any individual’s</u> 

#### Basis Functions

A set of functions that can be used to represent other functions in gen $f(x) = \sum_i c_i\cdot B_i(x)$ (so the Fourier series, polynomial series are sets of basis functions)

### Real-time Environment Lighting (& Global Illum)

#### Spherical Harmonics (SH)

A set of 2D basis functions $B_i(\omega)$ defined on the sphere

Analogous to Fourier series in 1D (same freq, function num = $2l + 1$, $l$ for order, first $n$ order has $2^n$) 

​	In this graph: the colors represents the values (represents pos/neg), the change of the color represent the frequency

​	The environment map describe a 2D function. This function projects to a basis function to obtain a value

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220319103159001.png" alt="image-20220319103159001" style="zoom:80%;" />

Each SH basis function $B_i(\omega)$ is associated with a (Legendre) polynomial

- **Projection**: obtaining the coefficients of each SH basis function (for every $\omega$ can find its projection on any basis func) 
  $$
  c_i=\int_{\Omega} f(\omega)B_i(\omega)\ \mathrm{d}\omega
  $$

- **Reconstruction**: restoring the original function using the (<u>trucated</u>) coefficients and basis functions

Usually pick the first several order to compute

> - SH is very similar to the Cartesian coordinate: In a 3D Cartesian coord, to represent a vector needs the projection of this vector on x, y, z (axis); In SH, the coefficients are the projections on basis function (~ x, y, z, …)
> - The x, y, z axis in the Cartesian are basically orthogonal, so every axis’s projection on the others’ is 0; similar in basis functions, a basis function’s projections on the others’ are also 0 (orthogonal property)
> - The product integer is actually dot product (in the expansion)

#### Prefiltered Env. Lighting

##### Recall: Prefiltering

**Prefiltering + single query = no filtering + multiple queries** (diffuse)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220319231200226.png" alt="image-20220319231200226" style="zoom:43%;" />

##### Analytic Irradiance Formula

<u>Diffuse</u> BRDF acts like a <u>low-pass filter</u> (relatively simple: smooth) 
$$
E_{\text{lm}} = A_l L_{\text{lm}} \ ;\quad \text{where } 
A_l = 2\pi \frac{(-1)^{\frac{l}{2}-1}}{(l+2)(l-1)} \left[\frac{l!}{2^l \left(\frac{l}{2}! \right)^2} \right],\ l \text{ even}
$$
The rendering equation: Lighting as a spherical function of the env map, BRDF as a smooth function defined on the sphere => The product integral (sum) of these 2 terms (use knowledge in SH) 

So the diffuse BRDF being regarded as a low-pass filter is a very good property because SH describes from low freq to high freq (usually only **3rd order** is enough, the projection as the graph shown below ($l$ as the order), due to very little high-freq contents)

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20220319234650679.png" alt="image-20220319234650679" style="zoom:40%;" />

> No matter how high freq the env lighting is (complex details in the env map), for a diffuse surface there will be little high freq content finally

##### 9 Parameter Approximation

| Exact Image                                                  | Order 0 (1 term)                                             | Order 1 (4 terms)                                            | Order 2 (9 terms)                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20220320000115072](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000115072.png) | ![image-20220320000120974](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000120974.png) | ![image-20220320000128221](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000128221.png) | ![image-20220320000134635](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000134635.png) |
| -                                                            | RMS error = 25%                                              | RMS error = 8%                                               | RMS error = 1% (sufficient)                                  |

Use 3 orders: for any illumination and diffuse materials, average error < 3%  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320000606520.png" alt="image-20220320000606520" style="zoom: 67%;" />

##### In RTR Shading with SH

$$
E(n) = n^t Mn
$$

Simple procedural rendering method (no texture)

- Requires only matrix-vector multiply and dot-product
- In software or NVIDIA vertex programming hardware

Widely used in games (AMPED for Xbox), movies (Pixar, Framestore CFC, …)

#### Precomputed Radiance Transfer (PRT)

> handles shadows and gi; but costs …

##### Rendering Under Environment Lighting

Usually write in the form of a triple product ($\vb{i/o}$ - incoming/view directions)
$$
L(\vb{o}) = \int _{\Omega} \underbrace{L(\vb{i})}_{\text{light}} \underbrace{V(\vb{i})}_{\text{vis}} \underbrace{\rho(\vb{i,o}) \max(0,\vb{n\cdot i})}_{\text{BRDF}}\ \mathrm{d} \vb{i}
$$

| Lighting $L(\vb{i})$                                         | Visibility $V(\vb{i})$                                       | BRDF $\rho(\vb{i,o}) \max(0,\vb{n\cdot i})$                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20220320112552070](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320112552070.png) | ![image-20220320112558296](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320112558296.png) | ![image-20220320112603594](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320112603594.png) |

For every shading point need these terms. For a 64x64 resolution pic: needs 6x64x64 times (multiplication of the 3 terms) for each point

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320114115600.png" alt="image-20220320114115600" style="zoom: 50%;" />

##### Basic Idea of PRT

> Precomputed Radiance Transfer for Real-Time Rendering in Dynamic, Low-Frequency Lighting Environments [Sloan 02]  

$$
L(\vb{o}) = \int _{\Omega} 
\underbrace{L(\vb{i})}_{\text{lighting}} \underbrace{V(\vb{i}) \rho(\vb{i,o}) \max(0,\vb{n\cdot i})}_{\text{light transport}}\ \mathrm{d} \vb{i}
$$

Suppose that in this scene everything other than lighting (represented as “light transport”) can be changed (camera can be changed either for diffuse)

- Approximate <u>lighting</u> using <u>basis functions</u> $L(\vb{i})\approx \sum l_i B_i(i)$ 
- **Precomputing** stage: compute <u>light transport</u>, and <u>project</u> to basis function space
- **Runtime** stage: <u>dot product</u> (diffuse) or <u>matrix-vector multiplication</u> (glossy)

##### Diffuse Case

Use $L(\vb{i})\approx \sum l_i \mathrm{B}_i(i)$ approximation where $l_i$ as the lighting coefficient and $B_i(\vb{i})$ as basis function
$$
\begin{aligned}
L(\vb{o}) &= \rho \int_{\Omega } L(\vb{i}) V(\vb{i}) \max(0, \vb{n\cdot i}) \ \mathrm{d}\vb{i }\\
&\approx \rho \sum l_i \underbrace{\int _{\Omega}  B_i(\vb{i}) V(\vb{i}) \max(0, \vb{n\cdot i}) \ \mathrm{d}\vb{i}}_{\text{precompute } \Rightarrow\ T_i}\\
&\approx \rho \sum l_iT_i
\end{aligned}
$$
The rendering computation becomes two vectors’ dot products

The split terms have another way to represent: $L(\omega_i)\approx \sum_p c_p B_p(\omega_i)$ or $T(\omega_i)\approx \sum_q c_q B_q(\omega_i)$ ($c_p$ - lighitng coeff; $c_q$ - light transport coeff)  ->  still $\mathcal{O}(n)$ (<u>orthonormal</u>, if and only if $p=q$, then $L_o \ne 0$) 
$$
L_o(\mathrm{p},\omega_o) = \sum_p \sum_q c_p c_q \int_{\Omega^+} B_p(\omega_i) B_q(\omega_i) \ \mathrm{d}\omega_i
$$
**Problem**: the <u>scene</u> cannot be changed (SH can compute the coefficients after the light sources rotate)

**Precomputation**

Light transport: (multiple times of bounces can also be regarded as a part of light transport to be p)
$$
T_i \approx\int _{\Omega}  B_i(\vb{i}) V(\vb{i}) \max(0, \vb{n\cdot i}) \ \mathrm{d}\vb{i}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320121803830.png" alt="image-20220320121803830" style="zoom: 50%;" />

**Run-time Rendering**
$$
L(\vb{o}) \approx \rho \sum l_i T_i
$$

- First, project the lighting to the basis to obtain $l_i$ 
- Or, rotate the lighting instead of re-projection
- Then, compute the dot product  

Real-time: easily implemented in shader  

##### Basis Functions

Good properties of SH

- orthonormal
  $$
  \int_{\Omega} B_{i}(\mathbf{i}) \cdot B_{j}(\mathbf{i})\ \mathrm{d} \mathbf{i}=\mathbf{1} \;\;(\mathbf{i}=\mathbf{j})\quad\quad 
  \int_{\Omega} B_{i}(\mathbf{i}) \cdot B_{j}(\mathbf{i})\ \mathrm{d} \mathbf{i}=\mathbf{0} \;\;(\mathbf{i}\ne\mathbf{j})
  $$

- simple projection/reconstruction

  - projection: $l_i = \int_{\Omega} L(\vb{i})\cdot B_i(\vb{i})\ \mathrm{d}\vb{i}$ 
  - reconstruction: $L(\vb{i}) \approx l_i B_i(\vb{i})$ 

- simple rotation

- simple convolution

- few basis functions: low freqs

Light approximation examples:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220320121232810.png" alt="image-20220320121232810" style="zoom:50%;" />

##### Glossy Case

Main difference from the diffuse case: The **BRDF** term (4D BRDF: inlet/outlet dir (2D polar coord x 2))
$$
%\begin{aligned}
L(\vb{o}) = \int_{\Omega } L(\vb{i}) V(\vb{i}) \rho(\vb{i,o}) \max(0, \vb{n\cdot i}) \ \mathrm{d}\vb{i }\approx  \sum l_iT_i(\vb{o}) \approx \sum\left(\sum l_i t_{ij}  \right) B_j(\vb{o})\\
T_i(\vb{o}) \approx \sum t_{ij}B_j(\vb{o})\qquad (t_{ij}\text{ for transport matrix})
%\end{aligned}
$$
Glossy objects have a special property: related to the view (=> explain $T_i$ still a function of $\vb{o}$ after projection)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220324230702454.png" alt="image-20220324230702454" style="zoom:80%;" />

**Rendering**: vector-matrix multiplication

**Costs**: storage of matrices and multiplication costs

**Time Complexity** 

- #SH basis: 9/<u>16</u>/25
- Diffuse rendering: at each point: <u>dot-product of size 16</u>
- Glossy rendering: at each point: <u>vector(16) * matrix(16*16)</u>

> If the frequency of the glossy object is very high (ie. tends to a mirror material): requires extremely high order of SH, which is practically impossible to be solved by PRT (not suitable)

##### Interreflections and Caustics 

2 bounces of interreflections and caustics (many times of self reflections, LSDE)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220324233126501.png" alt="image-20220324233126501" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220324233203211.png" alt="image-20220324233203211" style="zoom:50%;" />

> Example transport paths: All from L to E
>
> | LE        | LGE              | L(D\|G)\*E                                | LS\*(D\|G)\*E |
> | --------- | ---------------- | ----------------------------------------- | ------------- |
> | light-eye | light-glossy-eye | light-diffuse or glossy several times-eye |               |

Every scene only the transport term to be precomputed. So runtime complexity independent on the transport term.

#### Summary

##### Limitations  

- Low-frequency
  - Due to the nature of SH
- Dynamic lighting, but static scene/material
  - Changing scene/material invalidates precomputed light transport
- Big precomputation data  

##### Follow Up Works

- More basis functions
- Dot product => triple products
- Static scene => dynamic scene
- Fix material => dynamic material
- Other effects: translucent, hair, …
- Precomputation => analytic computation  

#### More Basis Functions

- Sperical Harmonics (SH)
- Wavelet
- Zonal Harmonics
- Spherical Gaussian (SG)
- Piecewise Constant

##### Wavelet

2D Haar wavelet

![image-20220324235904644](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220324235904644.png)

- **Projection**
  - Wavelet transformation
  - Retain a small num of non-zero coefficients

- A non-linear approximation
- **Advantage**: All-freq representation

Non-linear wavelet light approximation: For a function on a 2D sphere => **Cube map** (wavelet transform)

![image-20220325000300324](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325000300324.png) ![image-20220325000349150](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325000349150.png)

On every face: split to 4 squares, the low-freq content into the upper-left square and left the high-freq contents in the others => iterations

iteratively transform and remain => strong compression but good results

> JPEG format image compression uses the similar technique DCT (discrete cosine transform)

**Result**: SH (low freq only) vs Wavelet (all freq, including high freq shadow)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325001012055.png" alt="image-20220325001012055" style="zoom: 67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325001017028.png" alt="image-20220325001017028" style="zoom: 67%;" />

**Limitation**: doesn’t support high speed <u>rotations</u> (while SH is good for rotations)



# Lecture 7-10 Real-time Global Illumination

## Introduction

### Global Illumination (GI)

Global illumination is important but complex: many complex reflections …

> [Ritschel et al., The State of the Art in Interactive Global Illumination]  
>
> ![image-20220325113950646](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325113950646.png)

In RTR, people seek <u>simple</u> and <u>fast</u> solutions to **one bounce indirect illumination** (2 bounces in total)

> From [GAMES101 - Lecture 16](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#introducing-global-illumination)
>
> Any directly lit surface (in the graph below: Q point) will act as a <u>light source</u> again (secondary light source)
>
> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325114256369.png" alt="image-20220325114256369" style="zoom: 50%;" />

Direct illumination vs one-bounce global illumination (dir+indir) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325114723893.png" alt="image-20220325114723893" style="zoom: 37%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325114742514.png" alt="image-20220325114742514" style="zoom:37%;" />

### Key

What are needed to illuminate any point $p$ with indirect illum:

- Q1: Surface patches directly lit: classic shadow mapping, each pixel on the shadow map is a small surface patch
- Q2: Contribution from each surface patch to $p$, then sum up all the patches’ contributions (each surface patch is like an area light) -> area light rendering equation

## GI in 3D

### Reflective Shadow Maps (RSM)

>  The classic shadow map is a perfect solution to the first problem (which surface patches are directly lit)

The exact outgoing radiance for each pixel is known but only for direction to the camera.

**Assumptions**

- Any reflector is <u>diffuse</u> (only secondary light sources, no need to assume receptors are also diffuse)
- Outgoing radiance is <u>uniform</u> toward all directions

#### Recall: Light Measurements of Interest

> [GAMES101 - Lecture 16 (Radiometry)](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES101.html#radiant-intensity)

| ![image-20220325155056179](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325155056179.png) | ![](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219163300389.png) | <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325155539052.png" alt="image-20220325155539052" style="zoom:77%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Light Emitted From A Source                                  | Light Falling On A Surface                                   | Light Traveling Along A Ray                                  |
| **Radiant Intensity**                                        | **Irradiance**                                               | **Radiance**                                                 |

#### Reflective Shadow Maps

> Q2: An itegration over the solid angle covered by the patch contribute from each surface patch to $p$; can be converted to the integration on the area of the patch (usually use the center point to approximate)

![image-20220325160017001](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220325160017001.png)

$$
\begin{aligned} L_{o}\left(\mathrm{p}, \omega_{o}\right) &=\int_{\Omega_{\mathrm{patch}}} L_{i}\left(\mathrm{p}, \omega_{i}\right) V\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \\ 
&=\int_{A_{\mathrm{patch}}} L_{i}(\mathrm{q} \rightarrow \mathrm{p}) V\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \mathrm{q} \rightarrow \mathrm{p}, \omega_{o}\right) \frac{\cos \theta_{p} \cos \theta_{q}}{\|q-p\|^{2}} \mathrm{~d} A \end{aligned}
$$
For a diffuse reflective patch

- $f_r = \rho / \pi$
- $L_i = f_r \cdot \Phi / \mathrm{d}A$ ($\Phi$ - the incident flux or energy)

$$
\Rightarrow  
E_{p}(x, n)=\Phi_{p} \frac{\max \left\{0,\left\langle n_{p} \mid x-x_{p}\right\rangle\right\} \max \left\{0,\left\langle n \mid x_{p}-x\right\rangle\right\}}{\left\|x-x_{p}\right\|^{4}}
$$

**Problem**:

- Every secondary light source’s contribution to all the possible shading points ($n^2$ problem, cannot be solved by one RSM) -> no solution

#### Accelaration

-> Not all pixels in RSM can distribute: <u>visibility</u> (difficult to deal with) / <u>orientation</u> / <u>distance</u> 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220326111719078.png" alt="image-20220326111719078" style="zoom: 33%;" />

**Idea**: Use the projection of the shading point on the shadow map to find the surrounding possible contributing points

**Accelaration**:

- In theory, all pixels in the shadow map can contribute to $p$ 
- To decrease the number
- Hint: Steps 1 and 3 in PCSS (blocker search -> arbitrary sampling)

Sampling to rescure (for any shading point, the number decreases to ~400 from 512^2^)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220326112106334.png" alt="image-20220326112106334" style="zoom:50%;" />

#### Content

Stuff that recorded in an RSM: <u>Depth</u>, <u>world coordinate</u>, <u>normal</u>, <u>flux</u>, etc.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220326112855636.png" alt="image-20220326112855636" style="zoom:80%;" />

#### Usage

Often used for <u>flashlights</u> in video games, because only need to shade a small area, low res (Gears of War 4, Uncharted 4, TLoU, etc.)

#### Properties

- Pros
  - Easy to implement
- Cons
  - Performance scales linearly w/ #lights (primary light sources)
  - No visibility check for indirect illumination (look fake)
  - Many assumptions: diffuse reflections, depths as distance, etc.
  - Sampling rate / quality tradeoff

RSM is hardware-accelarate version (rasterize version) of <u>Virtual Point Light (VPL)</u> method (belong to Instant Radiocity (IR))

Shadow mapping is a image-space method (RSM as well). But RSM independs on the influence whether the final camera pass is visible & LPV introduced later is based on RSM => can be regarded as a 3D-space method

### Light Propagation Volumes (LPV)

> First introduced in CryEngine 3 (Crysis series): <u>fast performance</u> and <u>good quality</u>

#### Ideas

Key Problem: Query the radiance from any direction at any shading point

Key Idea: Radiance travels in a <u>straight line</u> and does <u>not change</u> (similar in path tracing)

Key Solution: Use a 3D grid to propagate radiance from directly illuminated surfaces to anywhere else

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331104755219.png" alt="image-20220331104755219" style="zoom:40%;" />

#### Steps

1. <u>Generation</u> of radiance point set scene representation

   - Find directly lit surfaces
   - Simply applying RSM would suffice
   - May use a reduced set of diffuse surface patches (virtual light sources)

2. <u>Injection</u> of point cloud of virtual light sources into radiance volume

   - Pre-subdivide the scene into a 3D grid (in industry, usually apply a texture (may in 3D))

     > Usually the grid number will be an order of magnitude less than the total pixel num

   - For each grid cell, find enclosed virtual light sources

   - Sum up their directional radiance distribution

   - Project to first 2 orders of SH (4 in total)

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331105212486.png" alt="image-20220331105212486" style="zoom: 50%;" />

3. Volumetric radiance <u>propagation</u> 

   - For each grid cell, collect the radiance received from each of its 6 faces (in 3D space)
   - Sum up and again use SH to represent
   - Repeat this propagation several times till the volume becomes stable (iterations for 4-5 times)

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331105417328.png" alt="image-20220331105417328" style="zoom:40%;" />

4. <u>Scene lighting</u> with final light propagation volume (<u>Rendering</u>)

   - For any shading point, find the grid cell it is located in 
   - Grab the incident radiance in the grid cell (from all directions)
   - Shade

#### Problems

- **Light leaking**: too large grids (LPV (left) / Ref (right)) => smaller grids (too much storage required) => adaptive grids (cascading)

  ![image-20220331105826858](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331105826858.png)

  > For example, the light source of point $p$ can never lighten the right side of the wall (however if use grids and suppose the light sources are at the center, the right side of the wall will be lighten)
  >
  > <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331105548666.png" alt="image-20220331105548666" style="zoom:33%;" />

### Voxel Global Illumination (VXGI)

#### Ideas

Still a <u>two-pass</u> algorithm

Differences with RSM:

- Directly illuminated pixels -> (hierarchical) voxels (like MineCraft/Lego)

- Sampling on RSM -> tracing reflected cones in 3D (Note the inaccuracy in sampling RSM)

  > For every shading point do a “cone-tracing”

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331110704641.png" alt="image-20220331110704641" style="zoom:50%;" />

#### Steps

1. Voxelize the entire scene
2. Build a hierarchy

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331110958946.png" alt="image-20220331110958946" style="zoom:50%;" />

- **Pass 1 from the light**

  - Store the <u>incident</u> and <u>normal</u> distribution in each voxel 

    (> according to material to compute the distribution of the radiance)

  - Update on the hierarchy

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331111141023.png" alt="image-20220331111141023" style="zoom:50%;" />

- **Pass 2 from the camera**

  - For glossy surfaces, trace 1 cone toward the reflected direction

  - Query the hierarchy based on the (growing) size of the cone

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331111520363.png" alt="image-20220331111520363" style="zoom:33%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331111530861.png" alt="image-20220331111530861" style="zoom:38%;" />

  - For diffuse, trace several cones (e.g., 8)

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220331111719273.png" alt="image-20220331111719273" style="zoom:50%;" /> 

#### Result

Pretty good results, close to ray tracing

(but still pretty huge costs, may need pre-compute, too complex)

## GI in Screen Space

>  **Screen Space**: Use information only from the “screen”, i.e. <u>post processing</u> on existing renderi

### Screen Space Ambient Occlusion (SSAO)

> First introduced by Crytek as well

#### Ideas

**Why AO**: 

- Cheap to implement
- But enhances the sense of relative positions (by contact shadows)

**Definition** of SSAO: An <u>approximation</u> of <u>global illumination</u> in <u>screen space</u>

Key Ideas:

1. We don’t know the incident indirect lighting

   -> assume it is <u>constant</u> (for all shading points from all directions) (similar to Blinn Phong’s ambient)

   > Not accurate -> modify in SSDO

2. Considering <u>different visibility</u> (towards all directions) at different shading points (also known as “skylight”)

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220402223502217.png" alt="image-20220402223502217" style="zoom:50%;" />

3. Assuming <u>diffuse</u> materials

Ambient Occlusion:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220402223808937.png" alt="image-20220402223808937" style="zoom:50%;" />

#### Theory of AO

- Starts from the rendering equation:
  $$
  L_{o}\left(\mathrm{p}, \omega_{o}\right)=
  %L_{e}\left(\mathrm{p}, \omega_{o}\right)+
  \int_{\Omega^+}  
  L_{i}\left(\mathrm{p}, \omega_{i}\right)\
  f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\
  V(\mathrm{p},\omega_i)\cos\theta_i
  \ \mathrm{d} \omega_{i}
  $$

- Again, applying “the RTR approximation/equation”
  $$
  \int_{\Omega} f(x) g(x)\ \mathrm{d} x \approx \frac{\int_{\Omega_G} f(x)\ \mathrm{d} x}{\int_{\Omega_G}\ \mathrm{d} x} \cdot \int_{\Omega} g(x)\ \mathrm{d} x
  $$

- Separating the visibility term
  $$
  L_{o}^{\text{indir}}\left(\mathrm{p}, \omega_{o}\right)\approx
  \underbrace{
  \frac{\int_{\Omega^+ }  V(\mathrm{p},\omega_i)\cos\theta_i\ \mathrm{d}\omega_i}{\int_{\Omega^+}\cos\theta_i\ \mathrm{d}\omega_i}
  }_{k_A = (\int_{\Omega^+ }  V(\mathrm{p},\omega_i)\cos\theta_i\ \mathrm{d}\omega_i)/\pi}
  
  \cdot
  
  \underbrace{
  \int_{\Omega^+}  
  L_{i}^{\text{indir}}\left(\mathrm{p}, \omega_{i}\right)\
  f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\
  \cos\theta_i
  \ \mathrm{d} \omega_{i}}_
  {L_i^{\text{indir}}(p)\cdot \frac{\rho}{\pi}\cdot \pi = L_i^{\text{indir}}(p)\cdot \rho}
  $$

  - the first section $k_A$ can be regarded as the weighted-averaged visibility $\overline{V}$ from all directions
  - the second section: the radiance and the BRDF are assumed as const for AO

- A deeper understanding 1 (take the average of one term of the support of $G$)
  $$
  \int_{\Omega} f(x) g(x)\ \mathrm{d} x \approx \frac{\int_{\Omega} f(x)\ \mathrm{d} x}{\int_{\Omega}\ \mathrm{d} x} \cdot \int_{\Omega} g(x)\ \mathrm{d} x
  = \overline{f(x)} \cdot \int_{\Omega} g(x)\ \mathrm{d}x
  $$
  
  In AO, the approximation is <u>accurate</u> (const $G = L\cdot f_r$)
  
- A deeper understanding 2 (why can take the cosine term with $\mathrm{d}\omega_i$) 

  Introducing the <u>projected solid angle</u> $\mathrm{d}x_{\perp} = \cos\theta_{i}\ \mathrm{d}\omega_i$ (unit hemisphere -> unit disk)

  Integration of the projected solid angle == area of the unit disk == $\pi$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403152540545.png" alt="image-20220403152540545" style="zoom:40%;" />

- A much simpler understanding

  - Uniform incident lighting $L_i$ = const

  - Diffuse BRDF $-f_r = \rho/\pi$ also const

  - Therefore take both out of the integral
    $$
    \begin{aligned} L_{o}\left(\mathrm{p}, \omega_{o}\right) &=\int_{\Omega^{+}} L_{i}\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right) V\left(\mathrm{p}, \omega_{i}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \\
    &=\frac{\rho}{\pi} \cdot L_{i}(p) \cdot \int_{\Omega^{+}} V\left(\mathrm{p}, \omega_{i}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \end{aligned}
    $$


#### Compute the Occlusion Values

> compute $k_A(p)$ in real time

- In object space
  - Raycasting against geometry
  - Slow, requires simplifications and/or spatial data structures
  - Depends of scene complexity
- In screen space
  - Done in a post-rendering pass
  - No pre-processing required
  - Doesn’t dep on scene complexity
  - Simple
  - Not physically accurate

#### Screen Space Solutions

Using <u>Z-buffering</u> to approximate the scene geometry

Applying sampling in a sphere around each pixel to test if in occlusion (against buffer)

![image-20220403160331224](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403160331224.png)

If more than a half of the samples are inside, AO is applied (since we don’t have the normal texture/map, need to test both the upper hemishpere and lower one). The amount depending on the <u>ratio</u> fo samples that pass/fail depth test -> the <u>visibility</u> (directly apply as green/total)

However, some fails are incorrect (the one behind the red line: false occlusions). Samples are not weighted by $\cos\theta$ so not physically accurate but convincing

**Problems**: <u>False occlusions, halos</u> (some far scenes also become darker)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403162855543.png" alt="image-20220403162855543" style="zoom: 67%;" />

#### Choosing Samples

- More samples -> greater accuracy
- Many samples are needed for a good result, but for performance only about 16 samples are used
- Positions from randomized texture to avoid banding
- Noisy result, blurred with edge preserving blur (denoising)

#### Horizon Based Ambient Occlusion (HBAO)

> To obtain the normal information -> much better results 

Also in screen space

Approximates ray-tracing the depth buffer

Requires that the normal is known and only samples in a hemisphere

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403163254895.png" alt="image-20220403163254895" style="zoom:50%;" />

Also can compute the weighted results

Can consider the distance (hemisphere) to generate shadows

Results (SSAO vs HBAO)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403163629183.png" alt="image-20220403163629183" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220403163701742.png" alt="image-20220403163701742" style="zoom:50%;" /> 

### Screen Space Directional Occlusion (SSDO)

#### Ideas

- An improvement over SSAO
- Considering (more) actual indirection illumination

Key Ideas:

- Why have to assume uniform incident indirect lighting 
- Some information of indirect lighting is already known (secondary light sources)
- Similar to the idea of RSM

#### SSDO

- SSDO exploits the rendered direct illumination

  - Not from an RSM, but from the <u>camera</u>

    > Good results in color reflections

    ![image-20220404161857258](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404161857258.png)

- Very similar to path tracing

  - At shading point $p$, shoot a random ray
  - If it doesn’t hit an obstacle, direct illum
  - If it hits one, indirect illum

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404165029041.png" alt="image-20220404165029041" style="zoom:50%;" />

- Comparison between SSAO and SSDO:

  - AO: indir illum (red) + no indir illum (orange)
  - DO: no indir illum (red) + indir illum (orange) (same as path tracing)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404165342373.png" alt="image-20220404165342373" style="zoom:67%;" />

- Consider unoccluded and occluded directions separately
  $$
  L^{\mathrm{dir}}_{o} (\mathrm{p},\omega_o) = 
  \int_{\Omega^+,V=1} L_i^{\mathrm{dir}}(\mathrm{p},\omega_i) f_r(\mathrm{p},\omega_i,\omega_o) \cos\theta_i\,\mathrm{d}\omega_i\\
  L^{\mathrm{indir}}_{o} (\mathrm{p},\omega_o) = 
  \int_{\Omega^+,V=0} L_i^{\mathrm{indir}}(\mathrm{p},\omega_i) f_r(\mathrm{p},\omega_i,\omega_o) \cos\theta_i\,\mathrm{d}\omega_i
  $$

  > Indir illum from a pixel (patch) derived last lectures

- Similar to HBAO, test <u>samples</u>‘ depths in local hemispheres (known normal)

  > - First two images: A, B, D are blocked; C unoccluded => ABD can provide indir illum for P
  > - The last image: A is not blocked in sampling but actually occluded by a floating sphere; B is blocked in the view of P, but actually not occluded in the camera view

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404170607055.png" alt="image-20220404170607055" style="zoom: 80%;" />

#### Results & Problems

- Quality: still fast but closer to offline rendering

- Issues:

  - GI in <u>short range</u> as well

    > Due to the short range, the green reflection may not appear in the cube surface

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404172850391.png" alt="image-20220404172850391" style="zoom:50%;" />

  - <u>Visibility</u> (camera view, still not accurate)

  - Screen space issue: missing information from <u>unseen</u> surfaces (information loss)

    > The yellowish color shadow below disappears after the yellow slab turns

    ![image-20220404172527775](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220404172527775.png)

### Screen Space Reflection (SSR)

