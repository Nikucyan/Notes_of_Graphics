# GAMES101 - Overview of Computer Graphics

Lecturer: [Lingqi Yan](www.cs.ucsb.edu/~lingqi/ )

[Lecture Videos](https://www.bilibili.com/video/BV1X7411F744) | [Course Site](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html) | [HW Site](https://games-cn.org/forums/topic/allhw/)

---



# Linear Algebra (Lec. 2)

## Cross Product

- Check whether a point $P$ is on the left side of the line $\vec{AB}$: $\vec{AB} \times \vec{AP} > 0$ (left); $\vec{AB} \times \vec{AP} < 0$ (right) 

- Check whether a point $P$ is in a triangle $ΔABC$ or not: $\vec{AB} \times \vec{AP} > 0$ & $\vec{BC} \times \vec{BP} > 0$ & $\vec{CA} \times \vec{CP} > 0$ (inside)

  (Corner case: $\times$ product = 0, should be defined)

## Matrices

Product can be operated: $A \cdot B = (M\times N) (N\times P)$ (OK)

- Dot Product: $\vec{a} \cdot \vec{b} = \vec{a}^T \vec{b} = \begin{pmatrix} x_a & y_a & z_a\end{pmatrix} \cdot \begin{pmatrix} x_b \\ y_b \\ z_b\end{pmatrix} = \begin{pmatrix} x_ax_b & y_ay_b & z_az_b\end{pmatrix} $
- Cross Product: $\vec{a} \times \vec{b} = A^* b =\begin{pmatrix} 0 & -z_a & y_a \\ z_a & 0 & -x_a \\ -y_a & x_a & 0\end{pmatrix} \begin{pmatrix} x_b \\ y_b \\ z_b\end{pmatrix}$



# Transformation (Lec. 3-4)

## Transformation

1. **Scale Matrix**: Ratio s: 

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715010441369.png" alt="image-20210715010441369" style="zoom:50%;" />

   $$
   x' = sx;\;y' = sy\,\Leftrightarrow \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix}
   $$

2. **Reflection Matrix**:

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715010427112.png" alt="image-20210715010427112" style="zoom:50%;" />
   $$
   \left\{ \begin{array} xx'=-x \\ y' = y \end{array}\right. \Leftrightarrow \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix}
   $$
   
3. **Shear Matrix**:

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715010400583.png" alt="image-20210715010400583" style="zoom:50%;" />
   $$
   \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} 1 & a \\ 0 & 1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix}
   $$
   
4. **Rotate (2D) around (0, 0), counter-clock**

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715010533272.png" alt="image-20210715010533272" style="zoom:50%;" />
   $$
   R_{\theta} = \begin{bmatrix} \cos{\theta} & -\sin{\theta} \\ \sin{\theta} & \cos{\theta} \end{bmatrix}
   $$
   
5. **Translation** (Not linear transformation)

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715011601111.png" alt="image-20210715011601111" style="zoom:50%;" />
   $$
   \left\{ \begin{array} xx'= x + t_x \\ y' = y + t_y \end{array}\right. \Leftrightarrow \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
   $$

## Homogeneous Coordinates

(Add a third coordinate - w-coordinate)

- 2D point = (x, y, 1)^T^ 

- 2D vector = (x, y, 0)^T^  （平移不变性：Direction and magnitude only）

  $$\Rightarrow \begin{pmatrix} x' \\ y' \\ w'\end{pmatrix} = \begin{pmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1\end{pmatrix}\begin{pmatrix} x \\ y \\ 1\end{pmatrix} = \begin{pmatrix} x+t_x \\ y+t_y \\ 1\end{pmatrix}$$  In homog. coord. $\begin{pmatrix} x \\ y \\ w\end{pmatrix}$ in 2-D point: $\begin{pmatrix} x/w \\ y/w \\ 1\end{pmatrix} (w\neq 1)$

- Vector + Vector  = Vector (0 + 0)

  Point - Point = Vector (1 - 1)

  Point + Vector = Point (0 + 1)

  Point + Point = ?? (1 + 1) (Actually, Mid point)

## Affine Transformation

> 仿射

Affine map = linear map + translation

$\Rightarrow \begin{pmatrix} x' \\ y' \\ 1\end{pmatrix} = \begin{pmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1\end{pmatrix}\begin{pmatrix} x \\ y \\ 1\end{pmatrix}$  Dimension rises (costs)

## Inverse Transformation

$M\rightarrow M^{-1}$

## Composite Transform

**<u>Transformation Ordering Matters</u>**

Notice that the matrices are applied right to left（无交换律 但有结合律）

e.g.	$T_{(1,0)}\cdot R_{45} \cdot \begin{pmatrix} x' \\ y' \\ 1\end{pmatrix} $: $R_{45} \rightarrow T_{(1,0)}$;	$A_n(....(A_2(A_1(x))) = A_n \cdot ... \cdot A_2\cdot A_1\cdot \begin{pmatrix} x' \\ y' \\ 1\end{pmatrix}$

Pre-multiply n matrices to obtain a single matrix representing combined transform (for performance)

## Example: (Decomposing)

Rotate around point C:

Representation: $T(c)\cdot R\cdot T(-c)$	(Move to the original point - rotate - move back (-c))

## 3D Transformation

- 3D point = (x, y, z, 1)^T^ 

- 3D vector = (x, y, z, 0)^T^ 

  ~ (x/w, y/w, z/w)

### **Homogeneous Coordinates**  

$$\Rightarrow \begin{pmatrix} x' \\ y' \\ z' \\1 \end{pmatrix} = \begin{pmatrix} a & b & c & t_x \\ d & e & f & t_y \\ g & h & i & t_z \\ 0 & 0 & 0 & 1\end{pmatrix}\begin{pmatrix} x \\ y \\ z \\ 1\end{pmatrix}$$

**Order**: Linear transformation first, then translation（先线性变换，再平移）

### Rotate around Axis

x-axis: $(\vec{y}\times\vec{z} = \vec{x})$  $R_x(\alpha) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha & 0 \\ 0 & \sin\alpha & \cos\alpha & 0 \\ 0 & 0 & 0 & 1\end{pmatrix}$		z-axis: $(\vec{x}\times\vec{y} = \vec{z})$  $R_z(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha & 0 & 0 \\\sin\alpha & \cos\alpha & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1\end{pmatrix}$

<u>y-axis</u>: $(\vec{z}\times\vec{x} = \vec{y})$  $R_y(\alpha) = \begin{pmatrix} \cos\alpha & 0 & \sin\alpha & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\alpha & 0 & \cos\alpha & 0 \\ 0 & 0 & 0 & 1\end{pmatrix}$	（y 轴方向是相反的）

### Compose any 3D Rotations from R~x~, R~y~, R~z~

$\bold{R}_{xyz}(\alpha, \beta, \gamma) = \bold{R}_x(\alpha)\bold{R}_y(\beta)\bold{R}_z(\gamma)$

- **Rodrigues’ Rotation Formula**: By angle $\alpha$ around axis n (Euler Angles)
  $$
  \mathbf{R}(\mathbf{n}, \alpha)=\cos (\alpha) \mathbf{I}+(1-\cos (\alpha)) \mathbf{n} \mathbf{n}^{T}+\sin (\alpha) \underbrace{\left(\begin{array}{ccc}0 & -n_{z} & n_{y} \\ n_{z} & 0 & -n_{x} \\ -n_{y} & n_{x} & 0\end{array}\right)}_{\mathbf{N}}
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715020442614.png" alt="image-20210715020442614" style="zoom:67%;" />

## Viewing Transformations

### View / Camera Transformation（视图变换）

#### **MVP: Model-View Projection** 

(Model transformation: placing objects; View transformation: placing camera; Projection transformation)

Define camera first: 

- Position: $\vec{e}$ 
- Look-at / gaze direction: $\vec{g}$
- Up direction: $\vec{t}$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715110858353.png" alt="image-20210715110858353" style="zoom:67%;" /> 

Key Observation: If the camera and all objects move together, the “photo” will be the same => Transform to: the origin with up @ Y, look @ -Z

**Transformation Matrix** $M_{\mathrm{view}}$ in math: $M_{\mathrm{view}} = R_{\mathrm{view}}T_{\mathrm{view}}$  (Transformed to a std. coordinate)

$T_{\mathrm{view}} = \begin{bmatrix}1 & 0 & 0 & -x_e \\ 0 & 1 & 0 & -y_e \\ 0 & 0 & 1 & -z_e \\ 0 & 0 & 0 & 1 \end{bmatrix}$； $$R_{\text {view }}=\left[\begin{array}{cccc}x_{\hat{g} \times \hat{t}} & y_{\hat{g} \times \hat{t}} & z_{\hat{g} \times \hat{t}} & 0 \\ x_{t} & y_{t} & z_{t} & 0 \\ x_{-g} & y_{-g} & z_{-g} & 0 \\ 0 & 0 & 0 & 1\end{array}\right]$$

(Also known as model-view transformation)

### Projection Transformation

#### Orthographic Projection

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715112050817.png" alt="image-20210715112050817" style="zoom:67%;" /> 

Camera @ 0, Up @ Y, Look @ -Z. Translate & Scale the resulting rectangle to [-1, 1]^2^ 

(Looking at / along -Z is making near and far not intuitive (n > f). OpenGL uses left hand coordinates)

In general, map a cuboid [l, r] x [b, t] x [f, n] to canonical cube [-1, 1]^3^ 

**Transformation Matrix**: $$M_{\text {ortho }}=\left[\begin{array}{cccc}\frac{2}{r-l} & 0 & 0 & 0 \\ 0 & \frac{2}{t-b} & 0 & 0 \\ 0 & 0 & \frac{2}{n-f} & 0 \\ 0 & 0 & 0 & 1\end{array}\right]\left[\begin{array}{cccc}1 & 0 & 0 & -\frac{r+l}{2} \\ 0 & 1 & 0 & -\frac{t+b}{2} \\ 0 & 0 & 1 & -\frac{n+f}{2} \\ 0 & 0 & 0 & 1\end{array}\right]$$ (Translate center 0; Scale 2)

#### Perspective Projection (Most common)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715112333094.png" alt="image-20210715112333094" style="zoom:80%;" /> (Not parallel)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715121619362.png" alt="image-20210715121619362" style="zoom:67%;" /> (Similar triangle $y' = \frac{n}{z} y$)

**Process**: Frustum（视锥）- (n - n, f - f) - Cuboid - Orthographic Proj. (M~o~) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210715112553604.png" style="zoom: 50%;" /> 

（近平面不变，远平面中心不变) 

**In homogeneous coordinates**: 

$\left(\begin{array}{l}x \\ y \\ z \\ 1\end{array}\right) \Rightarrow\left(\begin{array}{c}n x / z \\ n y / z \\ \text { unknown } \\ 1\end{array}\right) \begin{gathered}\text { mult. } \\ \text { by z } \\ ==\end{gathered}\left(\begin{array}{c}n x \\ n y \\ \text { still unknown } \\ z\end{array}\right)$;  Replace z with n;  $M_{p\rightarrow o} = \begin{pmatrix}n & 0 & 0 & 0 \\ 0 & n & 0 & 0 \\ 0 & 0 & n+f & -nf \\ 0 & 0 & 1 & 0 \end{pmatrix}$ 

$M_p = M_o \cdot M_{p\rightarrow o}$

**Vertical Field of View (fovY)**	(Assuming symmetry: l = -r; b = -t)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210716002006747.png" alt="image-20210716002006747" style="zoom:50%;" /> 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210716002132476.png" alt="image-20210716002132476" style="zoom:50%;" /> 

$\tan\frac{\text{fovY}}{2} = \frac{t}{|n|}$; $\text{Aspect} = \frac{r}{t}$



# Rasterization (Lec. 5-7)

## Rasterize

Rasterize = Draw onto the screen

Define Screen Space: **Pixel**: (0, 0) - (Width - 1, Height - 1)（均匀小方块）; Centered @ (x + 0.5, y + 0.5) (Irrelavent to z)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210716003103499.png" alt="image-20210716003103499" style="zoom:50%;" /> 

$M_{\text{viewport}} = \begin{pmatrix} \frac{width}{2} & 0 & 0 & \frac{width}{2} \\ 0 & \frac{height}{2} & 0 & \frac{height}{2} \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$

### Sampling (Approximate a Triangle)

Rasterization as 2D sampling

For triangles, each pixel is whether **inside / outside** should be checked.

```c++
for (int x = 0; x < xmax; ++x)
    for (int y = 0; y < ymax; ++y)
        image[x][y] = inside(tri, x + 0.5, y + 0.5);
```

#### Evaluating `inside(tri, x, y)`

3 cross products: $\vec{AB}$, $\vec{BC}$, $\vec{CA}$  (Same symbol = inside)

(Want: All required points (pixels) inside the triangle)

**Edge cases:** covered by both tri. 1 and 2: Not process / specific

Instead of checking all pixels on the screen, using **incremental triangle traversal** / a **bouding box** (AABB) can sometimes be faster

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210716004224152.png" alt="image-20210716004224152" style="zoom:33%;" /> Suitable for thin and rotated triangles

### Artifacts 

(Error / Mistakes / Inaccuracies)

Jaggies (Too low sampling rate) / Noire / Wagon wheel effect (Signal changing too fast for sampling)

Idea: Blur - Sample

### Frequency Domain

In freq. domain: $f= \frac{1}{T}$ 

**Fourier Transform**: 
$$
f(x) = \frac{A}{2} + \frac{2A \cos(t\omega)}{\pi} - \frac{3A \cos(3t\omega)}{3\pi} + \frac{2A \cos(5t\omega)}{5\pi} - \frac{2A\cos(7t\omega)}{7\pi}+ ...
$$
(Higher freq. needs faster sampling. Or samples erroneously a low freq. signal)

**Filtering**: Getting rid of certain freq. contents (high / low / band / ...)

Filter out high freq. (Blur) - Low pass filter (e.g. Box function)

Theorem: Spatial domain  ·  Filter (Convolution Kernel)   =  ...

​						$\downarrow$ $\mathcal{F}$ 	$F(\omega)=\int_{-\infty}^{\infty} f(x) e^{-2 \pi i \omega x} d x$	  	$\uparrow$ Inverse Fourier Transform	

​				 Fourier domain  $\times$ 	Fourier filter					=  ...

- **Convolution**（卷积）: （加权平均滤波器）Product in spatial domain = Convolution in frequency domain

  Wider Kernal = Lower Frequency (Blurer)

**Sampling = Repeating Frequency Contents**

- **Aliasing** = Mixing freq. contents (sampling too slow)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210719012208952.png" alt="image-20210719012208952" style="zoom: 80%;" /> 

  Reduction: Increasing sampling rate; Antialiasing (Blur - Sample, Make F contents "narrower")

-  **Antialiasing ** = Limiting, then repeating

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210719012533952.png" alt="image-20210719012533952" style="zoom: 80%;" /> 

**Solution**: Convolve $f(x. y)$ by a 1-pixel box-blur & Sample pixel's center

### Antialiasing Techniques

**MSAA (Multisample Antialiasing)**

By supersampling (same effects as blur, then sampling) (1 pixel -> 2x2 / 4x4 & Average)

Cost: Increase the computation work (4x4 = 16 times) - Key pixel distribution

**FXAA (Fast Approximation AA)**

后期处理降锯齿，使用边界

**TAA (Temporal AA)** 

Use the previous one frame for AA

**Super Resolution (DLSS - Deep Learning Supersampling)**

## Visibility / Occlusion 

### Z (Depth) Buffering

深度缓存

**Point Algorithm**（由远到近 - 不好处理相互重叠的关系）

**Z-buffer**: Frame buffer for color values; z-buffer for depth (Smaller z - closer (darker); Larger z - farther (lighter))

**Algorithm** during Rasterization

```c
for (each triangle T)
	for (each sample (x, y, z) in T)
		if (z < zbuffer[x, y])		    // Closest sample so far （是否更新该像素，判断 z）
			framebuffer[x, y] = rgb;	// Update color
			zbuffer[x, y] = z;		    // Update depth （相同的可更可不更 - 可能为抖动）
		else
			;						  // Do nothing
```

Complexity: O(n) 

​	For n triangles, only check. No other relativity.

Order doesn't matter

## Supplement: Shadow Mapping

After Lec. 12

> An image-space algorithm, widely used for early animations and every 3D video game

Key idea: The point not in shadow must be seen both by the **light** and the **camera**

**Hard shadow**: Use **0 & 1 only** to represent in shadow or not 

​	(between 0 and 1 -> soft shadow: by multiple light sources / source size)

### Approaches

- Pass 1: Render from light (Rasterize the view with the light source)

  Depth image from light source

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173441462.png" alt="image-20211217173441462" style="zoom:67%;" />

- Pass 2A: Render from eye

  Standard image with depth from eye

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173552587.png" alt="image-20211217173552587" style="zoom:67%;" />

- Pass 2B: Project to light

  Project visible points in eye view back to light source  

  <u>Orange line</u>: (Reprojected) depths match for light and eye. **VISIBLE**  

  <u>Red line</u>: (Reprojected) depths from light and eye are not the same. **BLOCKED**!!  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173643580.png" alt="image-20211217173643580" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217173727667.png" alt="image-20211217173727667" style="zoom:67%;" />

### Problems with Shadow Maps

- Hard shadows (point lights only)
- Quality dep on shadow map resolution (gen problem with image-based techniques)
- Involves equality comparison of floating point depth values means issues of scales / bias / tolerance



# Shading (Lec. 7-10)

Apply a material to an object

## Shading Model (Blinn-Phong Reflectance Model)

Light reflected towards camera.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210721175148434.png" alt="image-20210721175148434" style="zoom:67%;" /> 

Viewer direction, v; Surface normal, n; Light direction, l (for each of many lights); Surface parameters (color, shininess, …)  

**Shading is local** - No shadows will be generated

### Diffuse Reflection

Light is scattered uniformly in all directions (Surface color is the same for all viewing direction)

#### Light Falloff (Beer Lambert's Law)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210721175802013.png" alt="image-20210721175802013" style="zoom:50%;" /> 

#### Lambertian (Diffuse) Shading 

Shading independent of view direction
$$
L_{d}=k_{d}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{l})
$$
($L_d$​ - diffusely reflected light; $k_d$​ - diffuse coefficient (color), $k_d = 0$: black (all absorbed), $1$: white; $(I/r^2)$​ - energy arrived at the shading point; $\max(0, \mathbf{n} \cdot \mathbf{l})$​ - energy received by the shading point​; irrelavent to $\hat{v}$)

### Specular

Intensity depends on view direction (Bright near mirror reflection direction)

$\hat{v}$ closes to mirror direction - half vector near normal

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210722001128103.png" alt="image-20210722001128103" style="zoom:50%;" />

$\bold{h} = \text{bisector} (\bold{v},\bold{l}) = \frac{\bold{v}+\bold{l}}{||\bold{v}+\bold{l}||}$​ - 半程向量

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210722001625078.png" alt="image-20210722001625078" style="zoom:50%;" />
$$
\begin{aligned}
L_{s}&=k_{s}\left(I / r^{2}\right) \max (0, \cos \alpha)^{p} \\
&=k_{s}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{h})^{p}
\end{aligned}
$$
($L_s$​ - specularly reflected light; $k_s$​ - specular coefficient; $p$ - narrows the reflection lobe)

$p$ 越大，反射高光面积越小；$k_s$​ 越大，高光越明显

 (Only use $\bold{R}$ and $\bold{l}$ (mirror) - Phong relax model)

### Ambient

Not depend on anything 

Fake approximation

$L_a = k_a I_a$​ 	($L_a$​ - reflected ambient light; $k_a$ - ambient coefficient)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210722003002721.png" alt="image-20210722003002721" style="zoom:67%;" /> 

### Summary

$$
\begin{aligned}
L &=L_{a}+L_{d}+L_{s} \\
&=k_{a} I_{a}+k_{d}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{l})+k_{s}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{h})^{p}
\end{aligned}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210722003122186.png" alt="image-20210722003122186" style="zoom:100%;" />

## Shading Frequencies

(Face / Vertex / Pixel)

### Shading Methods

- Shade each **triangle** (**flat shading**) - Not good for smooth surface
- Shade each **vertex** (**Gouraud shading**) - Interporate colors from vertices
- Shade each **pixel** (**Phong shading**) - Full shading model each pixel (Not the Blinn-Phong Reflectance Model)

 <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726162249468.png" alt="image-20210726162249468" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726162302011.png" alt="image-20210726162302011" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726162311304.png" alt="image-20210726162311304" style="zoom:50%;" /> 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726163228332.png" alt="image-20210726163228332" style="zoom:50%;" /> 

### Pre-Vertex Normal Vectors

Vertex normal - Average surrounding face normals	$N_v =\frac{\sum_i N_i}{||\sum_i N_i||}$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726163313740.png" alt="image-20210726163313740" style="zoom:50%;" /> 

~ Barycentric interpolation of vertex normals (Need to normalize)

## Graphics (Real-time Rendering) Pipeline

~ GPU

- Input: Model, View, Projection transforms
- Rasterization: Sampling triangle coverage
- Fragment Processing: Z-Buffering visibility test / Shading / Texture mapping

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726163726899.png" alt="image-20210726163726899" style="zoom:67%;" /> 

## Texture Mapping

纹理映射 - UV ((u,v) coordinate) ($u, v \in [0, 1]$)

### Barycentric Coordinates

### Interpolation Access Triangles

Specific values <u>@ vertices</u> / smoothly varing <u>across triangles</u>

#### Barycentric Coordinates

A coordinate system for triangles $(\alpha, \beta, \gamma)$​​​​ (every point could be represented in this form). Inside the triangle if all three coordinates are non-negative

The barycentric coordinate of $A$​ is:  $(\alpha, \beta, \gamma) = (1, 0, 0)$​ ;  $(x,y) = \alpha A + \beta B + \gamma C =A$​

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726174026150.png" alt="image-20210726174026150" style="zoom: 33%;" /> 	$(x,y) = \alpha A + \beta B + \gamma C$ ;  $\alpha + \beta + \gamma = 1$	$\alpha\leq 1,\, \beta\leq 1,\, \gamma \leq1$

Geometric viewpoint — proportional **areas**:  $\alpha = \frac{A_A}{A_A+A_B+A_C}$​​​ ;  $\beta = \frac{A_B}{A_A+A_B+A_C}$​​​ ;  $\gamma = \frac{A_C}{A_A+A_B+A_C}$​​​ 

**Centroid**:  $(\alpha, \beta, \gamma) = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ ;  $(x,y) = \frac{1}{3}A + \frac{1}{3} B + \frac{1}{3} C$	$(A_A = A_B = A_C = \frac{1}{3})$​

Express **any point**: 
$$
\begin{aligned}
\alpha &=\frac{-\left(x-x_{B}\right)\left(y_{C}-y_{B}\right)+\left(y-y_{B}\right)\left(x_{C}-x_{B}\right)}{-\left(x_{A}-x_{B}\right)\left(y_{C}-y_{B}\right)+\left(y_{A}-y_{B}\right)\left(x_{C}-x_{B}\right)} \\
\beta &=\frac{-\left(x-x_{C}\right)\left(y_{A}-y_{C}\right)+\left(y-y_{C}\right)\left(x_{A}-x_{C}\right)}{-\left(x_{B}-x_{C}\right)\left(y_{A}-y_{C}\right)+\left(y_{B}-y_{C}\right)\left(x_{A}-x_{C}\right)} \\
\gamma &=1-\alpha-\beta
\end{aligned}
$$
**Color** interpolation for every point: linear interpolation	$V = \alpha V_A + \beta V_B + \gamma V_C$ (could be any property)

Disadvantage: Not invariant under projection (depth matters)

(3D - Use 3D interpolation OK; 2D - rather than use projection)

### Apply: Diffuse Color

``` pseudocode
for each rasterized screen sample (x,y)		// Usually a pixel center
    (u,v) = evaluate texture coordinate at (x,y)	// Applying Barycentric coordinates
    texcolor = texture.sample(u,v);
	set sample’s color to texcolor;  	// Usually diffuse kd (Blinn-Phong)
```

### Texture Magnification (AA)

(used for insufficient resolution)

**Texel**: A pixel on a texture（纹理元素 / 纹素）

#### Easy Cases

- Nearest
- Bilinear (Interpolation) - Pretty good results at reasonable costs
- Bicubic (Interpolation) - Instead of 4 points (bilinear), using 16 (4x4) points for 1 lerp

##### Bilinear Magnification

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726224355412.png" alt="image-20210726224355412" style="zoom:50%;" /> 

- Linear Interpolation (1D):  $\mathrm{lerp} (x, v_0, v_1) = v_0 +x(v_1 - v_0)$​
- Helper Lerps:  $u_0 = \text{lerp}(s,u_{00}, u_{10})$ ;  $u_1 = \mathrm{lerp} (s, u_{01}, u_{11})$ (Horizontal)
- Final Vertical Lerp:  $f(x, y) = \mathrm{lerp} (t, u_0, u_1)$​ (Vertical)

Lerp - Linear Interpolation

#### Hard Cases

##### Problem of point sampling textures 

(can be solved by supersampling but costly)

Near: Jaggies (Minification / Downsampling for too high resolution) ; Far: Moire (Upsampling)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726225139847.png" alt="image-image-20210726225139847" style="zoom:100%;" /> 

##### Mipmap

Allowing (fast, approx., square) range queries

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726225937700.png" alt="image-20210726225937700" style="zoom:67%;" /> 

(Image Pyramid) "Mip Hierachy": level = D

- Estimate texture footprint using texture coordinates of neighboring screen samples

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726230157286.png" alt="image-20210726230157286" style="zoom:50%;" /> 

- $$
  D = \log_2 L \quad L=\max \left(\sqrt{\left(\frac{d u}{d x}\right)^{2}+\left(\frac{d v}{d x}\right)^{2}}, \sqrt{\left(\frac{d u}{d y}\right)^{2}+\left(\frac{d v}{d y}\right)^{2}}\right)
  $$

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726230307099.png" alt="image-20210726230307099" style="zoom:50%;" /> 层查询 - 后者为选择较大 L 是近似（UV)

Near - range quires in low D level Mipmap; Far - high D level 

​	If want more continuous results: interpolation between levels

##### Trilinear Interpolation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726230541800.png" alt="image-20210726230541800" style="zoom:67%;" /> 

Linear interpolation (lerp) based on continuous D value

##### Limitations

In far and complex region - Overblur (box)

- **Anisotropic filtering**（各向异性过滤）: look up axis-aligned rectangular zones（长方形区域搜寻）, but still have problems in diagonal footprints - Ripmap (need graphics memory, doesn't need high flops x3)
- **EWA filtering** (Time computing costs): use multiple look ups / weighted average. able to handle irregular footprints

###  Applications of Textures

#### Environmental Map

- **Environmental Map**: 有光从各个方向进入眼睛，所有光都有对应，用纹理映射环境光

  不记录深度，都认为是无限远

- **Environmental Lighting**: 环境光记录于球面 - 渲染时展开

  - Spherical EM (Problem: Prone to distortion (top and bottom))

  - Cube Map: A vector maps to cube point along that direction (6 square texture maps) - Less distortion but need direction for face computations

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210727001419299.png" alt="image-20210727001419299" style="zoom:33%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210727001444460.png" alt="image-20210727001444460" style="zoom:40%;" /> $(u, v) = (1, 1)$, $x = y = z$​; $(u, v) = (0, 0)$, $x = -y = -z$

    Right face has $x > |y|$ and $x > |z|$

#### Textures Affecting Shading

##### Bump / Normal Mapping

Adding surface details without adding more triangles (Perturb surface normal per pixel)

未改变几何 - 产生凹凸错觉

- Perturb the nromal (in flatland)
  - Original surface normal:  $n(p) = (0,1)$

  - Derivative @ p:  $p = c\cdot[h(p+1) - h(p)]$

  - Perturbed normal:  $n(p) = (-dp, 1)$​.normalized()

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210728175134284.png" alt="image-20210728175134284" style="zoom:67%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210728175103568.png" alt="image-20210728175103568" style="zoom:29.5%;" />

- Perturb the normal (in 3D) (local coordinates)
  - Original surface normal:  $n(p) = (0,0,1)$
  - Derivatives @ p:  $-\frac{dp}{du} = c_1  \cdot [h(\mathbf{u}+1) - h(\mathbf{u})]$ ;  $-\frac{dp}{dv} = c_2  \cdot [h(\mathbf{v}+1) - h(\mathbf{v})]$​
  - Perturbed normal:  $n = (-\frac{dp}{du}, -\frac{dp}{dv}, 1)$.normalized()

##### Displacement Mapping

Uses the same texture as in bumping mapping, but actually moves vertices

- 3D Procedural Noise + Solid Modeling  
- Provide Precomputed Shading
- 3D Texture and Volume Rendering

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210730162847529.png" alt="image-20210730162847529" style="zoom:67%;" /> 



# Geometry (Lec. 10-12)

## Representing Ways

- **Implicit**: algebraic surface / level sets（等高线法）/ constructive solid geometry (Boolean) / distance functions / fractal ... 

  (Don’t tell exact positions,tell spe. relationships. e.g. sphere: $x^2 + y^2 + z^2 = 1$, generally $f(x, y, z) = 0$)

  Hard for sampling but easy for test in / outside

  ​	**Distance functions**:

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803161758591.png" alt="image-20210803161758591.png" style="zoom:40%;" />

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803161825884.png" alt="image-20210803161825884" style="zoom:67%;" />

  ​	**Level Set**: (CT / MRI)

  ![image-20210803161910976](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803161910976.png)

- **Explicit**: point cloud / polygon-mesh (.obj) / subdivision / NURBS / ...

  (Generally, $f: \R^2 \rightarrow \R^3: (u, v) \rightarrow (x, y, z)$)
  
  Hard to test in / outside but easy to sample

## Curves

### Bézier Curves (Explicit)

- 3 pt. - quadratic Bézier 

  Connect $b_0$ - all $b_0^2$ - $b_2$​ (every $t$ in $[0,1]$)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803162714678.png" alt="image-20210803162714678" style="zoom:67%;" /> 

- 4 pt. - cubic Bézier 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803162911437.png" alt="image-20210803162911437" style="zoom:45%;" /> 

(De Casteljau’s Algorithm)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803163059425.png" alt="image-20210803163059425" style="zoom:50%;" /> 

#### Evaluating Bézier Curves 

**Algebraic Formula**  
$$
\begin{aligned}
\mathbf{b}_0^1 (t) =& (1-t) \mathbf{b}_0 +t \mathbf{b}_1\\
\mathbf{b}_1^1 (t) =& (1-t) \mathbf{b}_1 +t \mathbf{b}_2\\
\\
\mathbf{b}_0^2 (t) =& (1-t) \mathbf{b}_0^1 +t \mathbf{b}_1^1\\
\\
\Rightarrow \mathbf{b}_0^2 (t) =& (1-t)^2 \mathbf{b}_0 +2t(1-t) \mathbf{b}_1 + t^2 \mathbf{b}_2
\end{aligned}
$$
**Bernstein form** of a Bézier curve of order n:  
$$
\mathbf{b}^{n}(t)=\mathbf{b}_{0}^{n}(t)=\sum_{j=0}^{n} \mathbf{b}_{j} B_{j}^{n}(t)
$$
($\mathbf{b}^n$​​ - Bezier curve order n (vector polynomial degree n); $\mathbf{b}_j$​​ - Bezier control points (vector in $\R^N$​​); $B_j^n$​ - Bernstein polynomial (scalar polynomial of degree n))

where **Bernstein polynomials**:
$$
B_{i}^{n}(t)=\left(\begin{array}{l}n \\ i\end{array}\right) t^{i}(1-t)^{n-i}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210803164056094.png" alt="image-20210803164056094" style="zoom:45%;" />

At anywhere, sum of $b_i^3=1$.

### B-Splines

Could be adjusted partially

For C^2^ cont. -> NURBS (Non-Uniform Rational B-Splines) 

## Surfaces

- Bicubic Bézier: 4x4 array -> surface
- Mesh Operations: subdivision / simplification / regularization

### Bézier Surfaces

Extend Bézier curves to surfaces

**Bicubic Bézier Surface Patch** (4x4 array of control points)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217113416287.png" alt="image-20211217113416287" style="zoom:67%;" />

#### Evaluating Surface Position for Parameters (u,v) 

For bicubic Bézier surface patch:

Input: 4x4 control points; Output: 2D surface parameterized by $(u,v)$ in $[0,1]^2$  

##### Method: Separable 1D de Casteljau Algorithm  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217113608676.png" alt="image-20211217113608676" style="zoom: 45%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217114450300.png" alt="image-20211217114450300" style="zoom: 67%;" />

(u,v)-separable application of de Casteljau algorithm:

- Use de Casteljau to evaluate point u on each of the 4 curves in u -> gives 4 control points for “moving” the Bézier curve
- Use 1D de Casteljau to evaluate point v on the “moving” curve

### Mesh Operations: Geometry Processing

![image-20211217115232686](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217115232686.png)

#### Subdivision (Upsampling)

> **Increase resolution**

##### Loop Subdivision

Only for **triangle faces** 

Intro more triangle meshes: Split triangles -> Update old / new vertices differently (according to weights)

- For **new vertices**: Update to $\frac{3}{8} \cdot (A+B) + \frac{1}{8}\cdot (C+D)$ 
- For **old vertices** (e.g. degree 6 vertices): Update to $(1- nu) \cdot \text{original\_position} + u \cdot \text{neighbor\_position\_sum}$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217115523249.png" alt="image-20211217115523249" style="zoom: 25%;" /><img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211217115809222.png" alt="image-20211217115809222" style="zoom:25%;" />

##### Catmull-Clack Subdivision (General Mesh)   

Can be used for various types of faces                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217120725036.png" alt="image-20211217120725036" style="zoom: 50%;" />

After several steps of Catmull-Clack subdivision:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217131904295.png" alt="image-20211217131904295" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217131914303.png" alt="image-20211217131914303" style="zoom:72%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217131953950.png" alt="image-20211217131953950" style="zoom:83%;" />  

**Vertex Updating Rules** (Quad Mesh)

![image-20211217162608447](C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211217162637030.png) ![image-20211217162651822](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217162651822.png) ![](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217162608447.png)

- Face Point: $f = \frac{1}{4} (v_1 + v_2 +v_3+ v_4)$ 
- Edge Point: $e = \frac{1}{4} (v_1 + v_2 + f_1+f_2)$ 
- Vertex Point: $v = \frac{1}{16} (f_1 + f_2 +f_3 +f_4 + 2(m_1 + m_2 + m_3 + m_4) + 4p)$ 

using the average to make it smoother => similar to the method of blur

#### Simplification (Downsampling)

> Reduce the mesh amount to **enhance performance**

##### Edge Collapse

![image-20211217171519375](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217171519375.png)

**Quadric Error Metrics** (geometric error introduced)

New vertex should minimize its sum of **square distance** (L2 distance) to previously related triangle planes

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217171718104.png" alt="image-20211217171718104" style="zoom:67%;" />

-> could cause error changes of other edges => dynamically update the other affected (prior queue / …)

=> Greedy algorithm for optimization (easy for flat, hard to curvature)

#### Regularization (Same #triangles)

> **Improve quality**



# Ray Tracing (Lec. 13-16)

> **Rasterization** cannot handle global effects well (fast approximation with low quality, but real time). However, soft shadows / especially for light bounces more than once 
>
> **Ray Tracing** is usually off-line, accurate but very slow

## Ray Tracing Basis

### Light Rays

- Light travels in straight lines (not correct)
- Light rays do not “collide” with each other if they cross (still not correct)
- Light rays travel from the light sources to the eye (but the physics is invariant under path reversal - reciprocity)  

### Ray Casting

> **Pinhole Camera Model**  

- Generate an image by **casting one ray per pixel**

- Check for shadows by **sending a ray to the light**

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217210235216.png" alt="image-20211217210235216" style="zoom:67%;" />

#### Generating Eye Rays

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217210338484.png" alt="image-20211217210338484" style="zoom:67%;" />

#### Shading Pixels (Local Only)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217210537187.png" alt="image-20211217210537187" style="zoom:67%;" />

### Recursive (Whitted-Style) Ray Tracing

Ray can **travel through** some (transparent / semi-transparent) media, but need to consider the **energy loss**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217210627861.png" alt="image-20211217210627861" style="zoom:67%;" />

### Ray-Surface Intersection

#### Ray Equation

Ray is defined by its **origin** and a **direction** vector

($\vb{r}$ - point along ray’; $t$ - time; $\vb{o}$ - origin; $\vb{d}$ - normalized direction)
$$
\vb{r}(t)  = \vb{o} + t\vb{d}\qquad 0\le t< \infty
$$

#### Ray Intersection 

##### With Sphere

Sphere: $\vb{p}:\ (\vb{p-c})^2 - {R} = 0$ 

=> Solve for the intersection: $(\vb{o} + t\vb{d} - \vb{c})^2 - R^2 = 0$

![image-20211217211448934](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217211448934.png) <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217215517277.png" alt="image-20211217215517277" style="zoom:67%;" />

for a second order equation: $at^2 + bt + c =0$ 

where $a = \vb{d}\cdot \vb{d}$; $b= 2(\vb{o-c})\cdot \vb{d}$; $c = \vb{(o-c)\cdot (o-c)} - R^2$; $t = (-b\pm \sqrt{b^2 - 4ac})/2a$ 

##### With Implicit Surface

- General implicit surface: $\vb{p}:\ f(\vb{p}) = 0$
- Substitute ray equation: $f(\vb{o} + t\vb{d}) = 0$

Solve for real, positive roots:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217215718979.png" alt="image-20211217215718979" style="zoom: 33%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217215731943.png" alt="image-20211217215731943" style="zoom:33%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217215748891.png" style="zoom:33%;" />

- Sphere: $x^2 +y ^2 + z^2 = 1$
- Dohnut: $(R - \sqrt{x^2 + y^2})^2  + z^2 = r^2$
- 3D heart: $(x^2 + 9y^2/4 + z^2 - 1)^3 = x^2z^3 + 9y^2z^3/80$ 

##### With Triangle Mesh

Idea: Just intersect ray with each triangle (too slow) => can have 0, 1 intersections (ignoring multiple intersections)

-> Triangle is a plane => Ray-plane intersection + Test if inside triangle

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217221558145.png" alt="image-20211217221558145" style="zoom:50%;" />

- Plane Equation: $\vb{p}:\ \vb{(p-p')\cdot N} = 0$  (if $\vb{p}$ satisfies; it’s on the plane, $\vb{p}'$ - one point on the plane)

- Solve for intersection:

  Set $\vb{p = r}(t)$ and solve for $t$

  $(\vb{p-p'})\cdot \vb{N} = (\vb{o} + t\vb{d} - \vb{p'}) \cdot \vb{N}= 0$

  $t= \frac{(\vb{p'-o})\cdot \vb{N}}{\vb{d\cdot N}}$ ;  Check if $0\le t < \infty$ 

**Möller Trumbore Algorithm** (for triangle problem)

A faster approach, giving barycentric coordinate directly

$\vb{O} + t\vb{D} = (1- b_1- b_2) \vb{P}_0 + b_1 \vb{P}_1 + b_2 \vb{P}_2$  (where $b_1, b_2, (1-b_1-b_2)$ are barycentric coordinates)

RHS: represent any point in the plane (sum of the coefficients is 1)
$$
\left[\begin{array}{c}
t \\
b_{1} \\
b_{2}
\end{array}\right]=\frac{1}{{\mathbf{S}}_{1} \cdot {\mathbf{E}}_{1}}\left[\begin{array}{c}
{\mathbf{S}}_{2} \cdot {\mathbf{E}}_{2} \\
{\mathbf{S}}_{1} \cdot {\mathbf{S}} \\
{\mathbf{S}}_{2} \cdot {\mathbf{D}}
\end{array}\right]
\quad \text{where: }
\left\{\begin{aligned}
& \vb{E}_1 = \vb{P}_1 - \vb{P}_0\\
& \vb{E}_2 = \vb{P}_2 - \vb{P}_0\\
& \vb{S} = \vb{O-P}_0\\
& \vb{S}_1 = \vb{D\times E}_2\\
& \vb{S}_2 = \vb{S\times E}_1
\end{aligned}\right.
$$
Costs = 1 div, 27 mul, 17 add

Need to check if $t\in [0,\infty)$ & inside triangle as well ($b_1, b_2, (1-b_1-b_2) >0$)

## Accelerating Ray-Surface (Triangle) Intersection

Problem: Naive algorithm (for every triangle) = #pixels x #triangles (x #bounces) (very slow)

### Bounding Volumes

- Object fully contained in the volume
- If not hit the volume, it doesn’t hit the object
- Test BVol first then test object if it hits

#### Ray-AABB (Box) Intersection 

Understanding: **Box is the intersection of 3 pairs of slabs**

Specifically: **Axis-Aligned Bounding Box (AABB)** 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217224311518.png" alt="image-20211217224311518" style="zoom:50%;" />

2D example: Compute intersections with slabs and take intersection of $t_{\min}/t_{\max}$ intervals

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217224550654.png" alt="image-20211217224550654" style="zoom:67%;" />

- **Key ideas**: 

  - The ray **enters** the box **only when** it enters **all pairs of slabs** 
  - The ray **exits** the box **as long as** it exits **any pair of slabs** 

- For each pair, calculate the $t_{\min}$ & $t_{\max}$ (negative is OK) 

- For the 3D box: $t_{\text{enter}} = \max(t_{\min})$ & $t_{\text{exit}} = \min(t_{\max})$ 

- Results:

  - If $t_{\text{enter}}<t_{\text{exit}}$ => the ray **stays a while** in the box (must intersect)

    But ray is not a line (should check whether $t$ is negative for physical correctness)

  - If $t_{\text{exit}}<0$ => Box is “behind” the ray - No intersection

  - If $t_{\text{exit}} >= 0$ and $t_{\text{enter}} <0$ => Ray originally inside the box (have intersection)

- Summary: ray and AABB intersect iff: $t_{\text{enter}}< t_{\text{exit}} $ && $t_{\text{exit}}>=0$  

- Why Axis-Aligned:

  Slab perpendicular to x-axis: $t = \frac{\vb{p}'_x - \vb{o}_x}{\vb{d}_x}$ (1 sub, 1 div)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217225957828.png" alt="image-20211217225957828" style="zoom:50%;" /> 

### Acceleration with AABBs

> **Uniform Spatial Partitions  (Grids)**

#### Uniform Grids

##### Preprocess - Build Acceleration Grid

1. Find bounding box
2. Create grid
3. Store each object in overlaping cells

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217233859037.png" alt="image-20211217233859037" style="zoom:50%;" />

##### **Ray-Scene Intersection**

Find the first intersection (with the object) first. If found that the ray passes through a box in which there’s an obj -> test if intersects with the obj (NOT necessary) 

(Ray intersect with boxes - fast; with obj - slower)

Step through grid in ray traversal order (-> similar to the method of drawing a line in the rasterization pipeline)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217234016772.png" alt="image-20211217234016772" style="zoom:50%;" />

For each cell: test intersection with all objects stored at that cell

##### **Resolution**

Acceleration effects: One cell - no speedup; Too many cells - Inefficiency due to extraneous grid traversal  

Heuristic: #cell = C * #objs; where C ≈ 27 in 3D

##### Usage

- Work well on **large collections** of objects that are distributed **evenly** in size and space
- Fail in “Teapot in a stadium” problem (distributed not evenly, a lot of empty space)

#### Spatial Partitions

##### Discretize the Bounding Boxes (in 3D)

- **Oct-Tree**: Split the bounding boxes evenly (in 3D so 8 branches)

  Stopping criteria -> until empty boxes / sufficiently small 

- **KD-Tree**: Always split the bounding box in different direction after another (horizontally or vertically, e.g., Horizontally first, then vertically, then horizontally again …; In 3D: x -> y -> z -> x -> …) => distribution almost evenly (Similar properties as binary-tree)

- **BSP-Tree**: Not suitable for higher orders

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217234739898.png" alt="image-20211217234739898" style="zoom:50%;" />

##### KD-Tree

###### KD-Tree Pre-processing

Actually in every branch, the split should be applied every time (in the following plot, only show one of the splitted branch)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211217234829391.png" alt="image-20211217234829391" style="zoom:50%;" />

###### Data Structure for KD-Trees

- Internal nodes store:
  - Split axis: x-, y-, z-axis
  - Split position: coordinate of split plane along axis
  - Children: pointers to child nodes
  - **No obj are stored** in internal nodes
- Leaf nodes store: 
  - List of objs

###### Traversing a KD-Tree

For the outer box have intersection (internal node) -> consider the sub-nodes (split1)

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211218114736334.png" alt="image-20211218114736334" style="zoom:50%;" />

In sub-nodes: Leaf 1 (LHS) & internal (RHS) both have intersections -> internal node splits -> … -> intersections found (after traversing all nodes)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211218171638062.png" style="zoom:50%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211218172925534.png" alt="image-20211218171638062" style="zoom:50%;" />

When have intersection with a leaf node of the box, the ray should find intersection with **all the objects** in this node

**Problem**: 

- In 3D, KD-Tree is complex and not easy to be written (especially when triangles intersects with the box but not inside the box, …)
- Some objects can be counted several times (appears in various of leaf nodes)

#### Object Partitioning & Bounding Volume Hierarchy (BVH)

> According to objects other than space, very popular

##### Main Idea

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219003646638.png" alt="image-20211219003646638" style="zoom:50%;" />

Can introduce some spec. stopping criteria

**Property**: No triangle-box intersections / No obj appears in more than one boxes

**Problem**: Boxes can overlap -> more researches to split obj

##### Summary: Building BVHs

- Find bounding box
- Recursively split set of objects in 2 subsets
- **Recompute** the bounding box of the subsets
- Stop when necessary
- Store objects in each leaf nodes 

**Subdivide a node**:

- Choose a dimension to split
- Heuristic #1: Always choose the longest axis in node
- Heuristic #2: Split node at location of **median** objects (amount balance) -> quick splitting

**Termination criteria**:

- Heuristic: node contains few elements (e.g, < 5)

##### BVH Traversal (Pseudo Code)

``` c
Intersect(Ray ray, BVH node) {
    if (ray misses node.bbox) return;
    
    if (node is leaf node) 
        test intersection with all objs;
    	return closest intersection;
    
    hit1 = Intersect(ray, node.child1);
    hit2 = Intersect(ray, node.child2);	// recursive
    
    return the closer of hit1, hit2;
}
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219005028450.png" alt="image-20211219005028450" style="zoom: 67%;" />

## Radiometry 

> Measurement system and units for illumination
>
> Accurately measure the spatial properties of light
>
> ​	Terms: Radiant flux（辐射通量）, intensity, irradiance（辐射照度）, radiance（辐射亮度）
>
> ​	-> **Light emitted from a source (radiant intensity)** / **light falling on a surface (irradiance)** / **light traveling along a ray (radiance)**
>
> Perform lighting calculations in a physically correct manner

### Radiant Energy and Flux (Power)

- **Radiant energy** is the energy of electromagnetic radiation. Measured in units of joules: $Q\ [\mathrm{J = Joule}]$ 

- **Radiant flux (power)** is the energy emitted, reflected, transmitted or received per unit time: $\Phi = \frac{\mathrm{d}Q}{\mathrm{d}t}\ [\mathrm{W = Watt}]\ [\mathrm{lm = lumen}]^*$ 

  Flux - #photons flowing through a sensor in unit time

### Radiant Intensity

The radient (luminous) intensity is the power per unit <u>solid angle</u>（立体角） emitted by a point light source (candela is one of the SI units) (sr - solid radius)
$$
I(\omega) \equiv \frac{\mathrm{d}\Phi}{\mathrm{d}\omega} \quad  \mathrm{\left[\frac{W}{sr}\right]\ \left[\frac{lm}{sr} = cd = candela\right]}
$$

#### Angles and Solid Angles

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219105214185.png" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219105239000.png" style="zoom:50%;" />

- **Angle**: ratio of subtended arc length on circle to radius $\theta = l / r$ (circle has $2\pi$ **radians**)
- **Solid Angle**: ratio of subtended area on sphere to radius squared $\Omega = A/r^2$ (sphere has $4\pi$ **steradians**)

**Differential Solid Angles**: (The unit area of the differential retangular)
$$
\mathrm{d}A = (r\ \mathrm{d}\theta)(r \sin\theta\ \mathrm{d}\phi) = r^2\ \sin\theta\ \mathrm{d}\theta\ \mathrm{d}\phi\ ;\quad
\mathrm{d}\omega = \frac{\mathrm{d}A}{r^2} = \sin\theta\ \mathrm{d}\theta \ \mathrm{d}\phi\\
\text{Sphere: }S^2:\ \Omega = \int_{S^2} \mathrm{d}\omega = \int^{2\pi}_0 \int^{\pi}_0 \sin\theta\ \mathrm{d}\theta\ \mathrm{d}\phi = 4\pi
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219105542204.png" alt="image-20211219105542204" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219110246823.png" alt="image-20211219110246823" style="zoom:50%;" />

**$\omega$ as a Direction Vector**: Use $\omega$ to denote a dir vector (unit length)

#### Isotropic Point Source

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219111038437.png" alt="image-20211219111038437" style="zoom:50%;" />
$$
\Phi = \int_{S^2} I\ \mathrm{d}\omega = 4\pi I\ ; \quad I = \frac{\Phi}{4\pi}
$$

### Irradiance

The irradiance is the power per (perpendicular/projected) unit area incident on a surface point
$$
E(\vb{x}) \equiv \frac{\mathrm{d}\Phi(\vb{x})}{\mathrm{d}A}\quad \mathrm{\left[\frac{W}{m^2} \right]\ \left[\frac{lm}{m^2} = lux \right]}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219163300389.png" alt="image-20211219163300389" style="zoom:50%;" />

#### Lambert’s Cosine Law

> -> Remind Blinn-Phong model / Seasons occur

Irradiance at surface is proportional to cosine of angle between light dir and surface normal

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219163535729.png" alt="image-20211219163535729" style="zoom:67%;" />

- Top face of cube receives a certain amount of power: $E = \Phi / A$
- Top face of 60% rotated cube receives half power: $E = \Phi / 2A $
- In general, power per unit area is proportional to $\cos\theta = l\cdot n$: $E = \Phi \cos\theta / A$ 

#### Irradiance Falloff

> Recall -> Blinn-Phong’s Model

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210721175802013.png" alt="image-20210721175802013" style="zoom:50%;" />

Assume light is emitting power $\Phi$ in a uniform angular distribution. The irradiance decays, while the intensity is constant

- distance = 1: $E = \Phi / 4\pi$
- distance = r: $E' = \Phi/4\pi r^2 = E/r^2$ 

### Radiance

> Radiance is the fundamental field quantity that describes the distribution of light in an environment  
>
> - Quantity associated with a ray
> - All about computing radiance

The radiance (luminance) is the power emitted, reflected, transmitted or received by a surface, <u>per unit solid angle, per projected unit area</u> (two derivatives) -> the $\cos \theta$ accounts for projected surface area   ($\mathrm{p}$ - reflection point)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219164500412.png" alt="image-20211219164500412" style="zoom:50%;" />
$$
L(\mathrm{p},\omega) \equiv \frac{\mathrm{d}^2\Phi(\mathrm{p},\omega)}{\mathrm{d}\omega\ \mathrm{d}A\ \cos\theta}\quad
\mathrm{\left[\frac{W}{sr\ m^2} \right] \ \left[\frac{cd}{m^2} = \frac{lm}{sr \ m^2} = nit \right]} 
$$
-> **Recall**:

- Irradiance: power per projected unit area
- Intensity: power per solid angle

Then: Radiance: Irradiance per solid angle / Intensity per projected unit area

#### Incident Radiance

Incident radiance is the irradiance <u>per unit solid angle</u> **arriving** at the surface  

=> it is the light arriving at the surface along a given ray (point on surface and incident direction)  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219183620789.png" alt="image-20211219183620789" style="zoom:50%;" />
$$
L(\mathrm{p},\omega) = \frac{\mathrm{d}E}{\mathrm{d}\omega \cos\theta}
$$

#### Exiting Radiance

Exiting surface radiance is the intensity per unit projected area leaving the surface  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219164500412.png" alt="image-20211219164500412" style="zoom:50%;" />
$$
L(\mathrm{p},\omega) = \frac{\mathrm{d}I(\mathrm{p},\omega)}{\mathrm{d}A \cos\theta}
$$
e.g., for an area light it is the light emitted along a given ray (point on surface and exit direction).  

### Irradiance vs. Radiance

- Irradiance: total power received by area $\mathrm{d}A $ 
- Radiance: power received by area $\mathrm{d}A $  from “direction” $\mathrm{d}\omega$  

$$
\mathrm{d}E (\mathrm{p},\omega) = L_i(\mathrm{p}, \omega) \cos\theta\ \mathrm{d}\omega\\
E(\mathrm{p}) = \int_{H^2} L_i(\mathrm{p}, \omega) \cos\theta\ \mathrm{d}\omega
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219184153512.png" alt="image-20211219184153512" style="zoom:50%;" />

($H^2 $ - Unit Hemisphere)

## Bidirectional Reflectance Distribution Function (BRDF)

The function to indicate the property of reflection (smooth surface / rough surface)

### Reflection at a Point

**Radiance** from direction $\omega_i$ turns into the **power** $E$ that $\mathrm{d}A$ receives 
Then **power** $E$ will become the **radiance** to any other direction $\omega_o$   

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219184826949.png" alt="image-20211219184826927" style="zoom:50%;" />

- Differential irradiance **incoming**:  $\mathrm{d}E(\omega_i) = L(\omega_i) \cos\theta_i\ \mathrm{\omega_i}$ 
- Differential radiance **exiting** (due to $\mathrm{d}E(\omega_i)$):  $\mathrm{d}L_r(\omega_r)$ (will be a ratio)

### BRDF

The BRDF represents how much light is reflected into each outgoing dir $\omega_r$ from each incoming dir => the energy distribution in different dir

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219185514242.png" alt="image-20211219185514242" style="zoom: 50%;" />
$$
f_r (\omega_i \rightarrow \omega_r) = \frac{\mathrm{d}L_r(\omega_r)}{\mathrm{d}E_i(\omega_i)} = \frac{\mathrm{d}L_r(\omega_r)}{L_i(\omega_i)\cos\theta_i\ \mathrm{d}\omega_i} \quad \left[\mathrm{\frac{1}{sr}}\right]
$$
BRDF defines the **material**

### The Reflection Equation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219185922952.png" alt="image-20211219185922952" style="zoom:50%;" />
$$
L_r(\mathrm{p},\omega_r) = \int_{H^2} f_r (\mathrm{p}, \omega_i \rightarrow \omega_r ) L_i(\mathrm{p},\omega_i) \cos\theta\ \mathrm{d}\omega_i
$$
**Challenge: Recursive Equation**

Reflected radiance depends on incoming radiance; but incoming radiance depends on reflected radiance (at another point in the scene) <- the lights will not just bounce once

### The Rendering Equation

For objects can emit light -> adding an emission term to make the reflection function general ($\Omega^+$ and $H^2$ both represent the hemisphere)
$$
\underbrace{L_{o}\left(\mathrm{p}, \omega_{o}\right)}_{\begin{aligned}&\text{Reflected Light}\\ &(\text{Output image})\end{aligned}} =
\underbrace{L_{e}\left(\mathrm{p}, \omega_{o}\right)}_{\text{Emission}} +
\int_{\Omega^{+}} 
\underbrace{L_{i}\left(\mathrm{p}, \omega_{i}\right)}_{\begin{aligned}&\text{Incident Light}\\&(\text{from source}) \end{aligned}}
\underbrace{f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)}_{\text{BRDF}}
\underbrace{\left(n \cdot \omega_{i}\right)}_{\begin{aligned}&\text{Cosine of}\\ &\text{Inc Angle}\end{aligned}}
\ \mathrm{d} \omega_{i}
$$
Note: now assume that all dir are pointing outwards

#### Understanding the Rendering Equation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211219205847794.png" alt="image-20211219205847794" style="zoom:50%;" />

Yellow - Large light source; Blue - Surfaces (interreflection)

$L_r(x,\omega_r)$ & $L_r(x', -\omega_i)$ are unknown 
$$
L_{r}\left(x, \omega_{r}\right)=L_{e}\left(x, \omega_{r}\right)+\int_{\Omega} L_{r}\left(x^{\prime},-\omega_{i}\right) f\left(x, \omega_{i}, \omega_{r}\right) \cos \theta_{i}\ \mathrm{d} \omega_{i}
$$
is a **Fredholm Integral Equation** of second kind [extensively studied numerically] with canonical form

The kernel of equation replaced with the **Light Transport Operator**
$$
l(u) = e(u ) + \int l(v)\  \underbrace{K(u,v) \ \mathrm{d}v}_{\begin{aligned}&\text{kernel of equation}\end{aligned}}\\
\Rightarrow
L = E + KL
$$
=> solve the rendering equation: discretized to a simple matrix equation [or system of simultaneous linear equations: L, E are vectors, K is the light transport matrix] => Applying <u>binomial theorem</u> 
$$
\begin{aligned}
L &= E+KL\\
IL - KL &= E\\
L &= (I-K)^{-1} E\\
L & = (I + K + K^2 + K^3 + \cdots) E\\
L & = E + KE + K^2E + K^3E + \cdots

\end{aligned}
$$

#### Global Illumination

In ray tracing => **Global Illumination** (Higher order K)

- E: Emission directly from light sources -> Shading in **Rasterization**
- KE: Direct illumination on surfaces -> Shading in **Rasterization**
- K^2^E: Indirect illumination (One bounce) [mirrors, refraction] 
- K^3^E: Two bounce in direct illum.
- …

## Monte Carlo Path Tracing

### Probability Review

#### Random Variables

| Type         | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| $X$          | Random variable, represents a distribution of potential values |
| $X\sim p(x)$ | Probability density function (PDF), describing relative probability of a random process choosing value $x$ |

Uniform PDF: all values over a domain are equally likely  

#### Probabilities

n discrete values $x_i$ with probability $p_i$ 

Requirements of a probability distribution: $p_i>0$ && $\sum p_i = 1$ 

#### Expected Value

Expected value of $X$ (drawn from distribution above): $E [X] = \sum x_ip_i$ 

#### Continuous Case: Probability Density Function

Conditions on $p(x)$: $p(x)\ge 0$ and $\int p(x) \ \mathrm{d}x = 1$ 

Expected value: $E[X] = \int x p(x)\ \mathrm{d}x$ 

#### Function of a Random Value

$X\sim p(x)$ & $Y = f(X)$

=> Expected value: $E[Y] = E[f(x)] = \int f(x)\ p(x)\ \mathrm{d}x$ 

### Monte Carlo Integration

> We want to solve an integral but it can be too hard to solve analytically => **Monte Carlo** (numerical method)

Definite integral $\int_a^b f(x)\ \mathrm{d}x $ & Random variable $X_i \sim p(x)$ 

**Monte Carlo estimator**: 
$$
F_N = \frac{1}{N}\sum^N_{i=1} \frac{f(X_i)}{p(X_i)}\qquad X_i\sim p(x)
$$

- The more samples, the less variance
- Sample on $x$, integrate on $x$

### Path Tracing

#### Motivation: Whitted-Style RT

Whitted-style ray tracing

- Always perform specular reflections / refractions
- Stop bouncing at diffuse surfaces

**Problems**  => Not 100% reasonable

- The reflections can be divided into **glossy** (rough surface) and **mirror** (specular) => Witted-Style cannot present the glossy result
- **Diffuse material** can also reflect light (but to many directions), resulting in **color bleeding** (global illumination can present) -> Witted-Style doesn’t bounce light on diffuse surface (= direct illumination)

The Whitted-Style is not that correct -> But the rendering equation is correct

But involves: solving an <u>integral</u> over the hemisphere (=> Monte Carlo) and <u>recursive</u> execution

#### A Simple Monte Carlo Solution

Suppose we want to render **one pixel (point)** in the following scene for **direct illumination** only

For the **reflection equation**:
$$
L_{o}\left(p, \omega_{o}\right)=\int_{\Omega^{+}} L_{i}\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\left(n \cdot \omega_{i}\right)\ \mathrm{d} \omega_{i}
$$
Consider the <u>direction</u> as the random variable, want to compute the radiance at $\mathrm{p}$ towards the camera

**Monte Carlo integration**: 
$$
\int _a^b f(x)\ \mathrm{d}x \approx \frac{1}{N} \sum^N_{k=1} \frac{f(X_k)}{p(X_k)} \quad X_k \sim p(x)\ ;\quad 
\text{where } f(x) = L_i(\mathrm{p},\omega_i) f_r(\mathrm{p},\omega_i, \omega_o) (n\cdot \omega_i)
$$
and pdf: $p(\omega_i) = 1/2\pi$ (assuming uniformly sampling the hemisphere)

=> In general:
$$
\begin{aligned}
L_o(\mathrm{p},\omega_o) &= \int_{\Omega^{+}} L_{i}\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\left(n \cdot \omega_{i}\right)\ \mathrm{d} \omega_{i}\\
&\approx \frac{1}{N}\sum^N_{i=1} \frac{ L_{i}\left(\mathrm{p}, \omega_{i}\right) f_{r}\left(\mathrm{p}, \omega_{i}, \omega_{o}\right)\left(n \cdot \omega_{i}\right)}{p(\omega_i)}

\end{aligned}
$$
**Algorithm**: (Direct illumination)

``` c
shade(p, wo)
    Randomly choose N directions wi~pdf
    Lo = 0.0
    For each wi
    	Trace a ray r(p, wi)
    	If ray r hit the light
    		Lo += (1 / N) * L_i * f_r * cosine / pdf(wi)
    Return Lo
```

#### Introducing Global Illumination

Further step: what if a ray hits an object?

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220212833583.png" alt="image-20211220212833583" style="zoom:67%;" />

Q also refects light to P => = the dir illum at Q

New **Algorithm**:

```
shade(p, wo)
    Randomly choose N directions wi~pdf
    Lo = 0.0
    For each wi
    	Trace a ray r(p, wi)
    	If ray r hit the light
    		Lo += (1 / N) * L_i * f_r * cosine / pdf(wi)
    	Else If ray r hit a object at q
    		Lo += (1 / N) * shade(q, -wi) * f_r * cosine / pdf(wi)	// new step!
    Return Lo
```

**Problems**: 

- Explosion of #rays as #bounces go up: #rays = N^#bounces^ (#rays will not explode iff **N = 1**)

  From now on  => always assume that only **1 ray** is traced at each shading point => Actual Path Tracing (Shown in Path Tracing Algorithm)

  => Noisy if N = 1: **Tracing more paths** through **each pixel** and **average the radiance** (Shown in Ray Generation)

- Recursive in `shade()`: never stop

  but the light does not stop bouncing indeed / cutting #bounce == cutting energy

  => **Russian Roulette** (RR), shown below

**Path Tracing Algorithm**: (Recursive)

```
shade(p, wo)
	Randomly choose ONE direction wi~pdf(w)
	Trace a ray r(p, wi)
	If ray r hit the light
		Return L_i * f_r * cosine / pdf(wi)
	Else If ray r hit an object at q
		Return shade(q, -wi) * f_r * cosine / pdf(wi)
```

#### Ray Generation

**Tracing more paths** through **each pixel** and **average the radiance** (similar to ray casting in ray tracing)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220213906682.png" alt="image-20211220213906682" style="zoom:50%;" />

```
ray_generation(camPos, pixel)
	Uniformly choose N sample positions within the pixel
	pixel_radiance = 0.0
	For each sample in the pixel
		Shoot a ray r(camPos, cam_to_sample)
		If ray r hit the scene at p
			pixel_radiance += 1 / N * shade(p, sample_to_cam)
	Return pixel_radiance
```

#### Rusian Roulette (RR)

Basic Idea:

- With probability 0 < P < 1, you are fine
- With probability 1 - P, otherwise  

Previously, always shoot a ray at a shading point and get the result Lo

Suppose we can manually set a probability P (0 < P < 1): 

- **With P**, shoot a ray and return the shading result divided by P **(Lo/P)**
- **With 1-P**, don’t shoot a ray and get **0**

=> the expected value is still Lo: $E = P \cdot (L_o / P) + (1-P) * 0 = L_o$ (discrete)

**Algorithm with RR**: (real correct version)

``` c
shade (p, wo)
    Manually specify a probability P_RR
	Randomly select ksi in a uniform dist. in [0, 1]
	If (ksi > P_RR) return 0.0;

	Randomly choose ONE direction wi~pdf(w)
    Trace a ray r(p, wi)
    If ray r hit the light
		Return L_i * f_r * cosine / pdf(wi) / P_RR
	Else If ray r hit an object at q
		Return shade(q, -wi) * f_r * cosine / pdf(wi) / P_RR
```

**Samples Per Pixel** (SPP) => low (left): noisy

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220221554200.png" alt="image-20211220221554200" style="zoom:67%;" />

**Problem**:

- Not efficient

#### Sampling the Light

For area light sources, the ones with **bigger areas** have **higher probability to get hit** by the the “ray” if uniformly sample the hemisphere at the shading point => waste

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220221952108.png" alt="image-20211220221952108" style="zoom:67%;" />

Monte Carlo method allows any sampling method => make the waste less

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220222027952.png" alt="image-20211220222027952" style="zoom:50%;" />

Assume <u>uniformly sampling</u> on the light: $\text{PDF} = 1 / A$ (because $\int\text{PDF}\ \mathrm{d}A = 1$)

But the rendering equation <u>integrates on the solid angle</u>: $L_o = \int L_i f_r \cos\ \mathrm{d}\omega  $ => <u>integral on the light</u> since we sample on the light => make $\mathrm{d}A$ from $\mathrm{d}\omega$ 

Projected area on the unit sphere: $\mathrm{d}\omega = \frac{\mathrm{d}A\cos\theta'}{\|x'-x \|^2}$ (not $\theta$) 

Rewrite the **rendering equation** with $\mathrm{d}A$:
$$
\begin{aligned}
L_{o}\left(x, \omega_{o}\right) &=\int_{\Omega^{+}} L_{i}\left(x, \omega_{i}\right) f_{r}\left(x, \omega_{i}, \omega_{o}\right) \cos \theta\ \mathrm{d} \omega_{i} \\
&=\int_{A} L_{i}\left(x, \omega_{i}\right) f_{r}\left(x, \omega_{i}, \omega_{o}\right) \frac{\cos \theta \cos \theta^{\prime}}{\left\|x^{\prime}-x\right\|^{2}}\ \mathrm{d} A
\end{aligned}
$$
Now integrate on the light => $\text{PDF} = 1 / A$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211220222927005.png" alt="image-20211220222927005" style="zoom:33%;" />

Now we consider the radiance coming from 2 parts:

- **Light source** (direct, no need for RR)
- **Other reflections** (indirect, RR)

**Light Sampling Algorithm**:

``` python
shade(p, wo)
	# Contribution from the light source. (no RR)
	Uniformly sample the light at x’ (pdf_light = 1 / A)
	L_dir = L_i * f_r * cos θ * cos θ’ / |x’ - p|^2 / pdf_light
    
	# Contribution from other reflectors. (with RR) 
	L_indir = 0.0
	Test Russian Roulette with probability P_RR
	Uniformly sample the hemisphere toward wi (pdf_hemi = 1 / 2pi)
	Trace a ray r(p, wi)
	If ray r hit a non-emitting object at q
		L_indir = shade(q, -wi) * f_r * cos θ / pdf_hemi / P_RR
        
    Return L_dir + L_indir
```

Final: how to know if the sample on the light is not **block** or not?

``` python
# Contribution from the light source.
L_dir = 0.0
Uniformly sample the light at x’ (pdf_light = 1 / A)
Shoot a ray from p to x’
If the ray is not blocked in the middle
	L_dir = …
```



# Materials and Appearances (Lec. 17)

**Material  == BRDF**

## Materials

### Diffuse / Lambertian Material

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103243032.png" alt="image-20211223103243032" style="zoom: 33%;" />

Light is equally reflected in each output dir ($f_r = c$)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223102648064.png" alt="image-20211223102648064" style="zoom:80%;" />

Suppose the incident light is **uniform** (identical radiance). According to energy balace, incident and exiting radiance are the same.

$f_r = \frac{\rho}{\pi}$ ($\rho$ - albedo, diffuse rate (color)) 
$$
L_o(\omega_o) = \int_{H^2} f_r L_i(\omega_i) \cos \theta \ \mathrm{d}\omega_i =  f_r L_i\int_{H^2} \cancel{(\omega_i)} \cos \theta \ \mathrm{d}\omega_i = \pi f_r L_i
$$

### Glossy Material (BRDF) 

Air <-> copper / aluminum

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103308484.png" alt="image-20211223103308484" style="zoom:33%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103318631.png" alt="image-20211223103318631" style="zoom: 25%;" />

### Ideal Reflective / Refractive Material (BSDF*)

Air <-> water interface / glass interface (with partial abs)

-> the “S” in “BSDF” is for “Scatter” => both reflection and refraction are ok

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103409301.png" alt="image-20211223103409301" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103423187.png" alt="image-20211223103423187" style="zoom:25%;" />

## Reflection & Refraction

### Perfect Specular Reflection

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103644847.png" alt="image-20211223103644847" style="zoom:80%;" />	<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223103706268.png" alt="image-20211223103706268" style="zoom:80%;" />

Left: $\theta = \theta_o = \theta_ i$ ; Right: $\phi_o = (\phi_i + \pi)\ \mathrm{mod}\ 2\pi $ (top-down view)
$$
\omega  _o + \omega_i = 2\cos\theta\ \vb{n} = 2(\omega_i\cdot \vb{n}) \ \vb{n}	\ \Rightarrow \ 
\omega_o = -\omega_i + 2(\omega_i + \vb{n})\ \vb{n}
$$

### Specular Refraction

Light refracts when it enters a new medium

#### Snell’s Law

Transmitted angle depends on: Index of refraction (IOR) for incident and exiting ray

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223110612197.png" alt="image-20211223110612197" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223110630488.png" alt="image-20211223110630488" style="zoom:80%;" />  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223110759024.png" alt="image-20211223110759024" style="zoom:80%;" />

Left: $\eta _ i \sin\theta_i = \eta _t \sin \theta _t $; Right: $\varphi _t = \varphi_i \pm\pi$ (sim to reflection) 

*Diamond has a very high refraction rate, which means that the light will be refracted heavily inside the diamonds => shiny with various colors
$$
\eta _i \sin\theta_i = \eta_t \sin\theta_t\\ 
\cos\theta_t = \sqrt{1-\sin^2\theta _t} = \sqrt{1- \left(\frac{\eta_i}{\eta_t} \right)^2 \sin^2\theta_i} = \sqrt{1 - \left(\frac{\eta_i}{\eta_t} \right)^2 \left(1-\cos^2\theta_i \right)}
$$
Want a reasonable real number to have the refraction occurred (need $\cos\theta_t$ exist) => $\eta_i < \eta_t$

**Total internal reflection**: The internal media has a higher refraction rate than outside: $\eta _i / \eta_t > 1$ => Light incident on boundary from large enough angle will not exit medium  (e.g., Snell’s window / circle)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223111633375.png" alt="image-20211223111633375" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223111641582.png" alt="image-20211223111641582" style="zoom: 50%;" />

### Fresnel Reflection / Term

Reflectance depends on incident angle (and polarization of light)

e.g., reflectance increases with grazing angle  

- **Dielectric** ($\eta = 1.5$) -> the visible angle may be limited

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223111940911.png" alt="image-20211223111940911" style="zoom: 67%;" /> 

- **Conductor** 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223112049382.png" alt="image-20211223112049382" style="zoom:67%;" /> 

**Formulae**: 

**Accurate**: (considering the polarization)
$$
R_{\text{eff}} = \frac{1}{2} (R_{\mathrm{s}} + R_{\mathrm{p}})\quad
\left\{\begin{aligned}
R_{\mathrm{s}}=\left|\frac{n_{1} \cos \theta_{\mathrm{i}}-n_{2} \cos \theta_{\mathrm{t}}}{n_{1} \cos \theta_{\mathrm{i}}+n_{2} \cos \theta_{\mathrm{t}}}\right|^{2}=\left|\frac{n_{1} \cos \theta_{\mathrm{i}}-n_{2} \sqrt{1-\left(\frac{n_{1}}{n_{2}} \sin \theta_{\mathrm{i}}\right)^{2}}}{n_{1} \cos \theta_{\mathrm{i}}+n_{2} \sqrt{1-\left(\frac{n_{1}}{n_{2}} \sin \theta_{\mathrm{i}}\right)^{2}}}\right|^{2}\\
R_{\mathrm{p}}=\left|\frac{n_{1} \cos \theta_{\mathrm{t}}-n_{2} \cos \theta_{\mathrm{i}}}{n_{1} \cos \theta_{\mathrm{t}}+n_{2} \cos \theta_{\mathrm{i}}}\right|^{2}=\left|\frac{n_{1} \sqrt{1-\left(\frac{n_{1}}{n_{2}} \sin \theta_{\mathrm{i}}\right)^{2}}-n_{2} \cos \theta_{\mathrm{i}}}{n_{1} \sqrt{1-\left(\frac{n_{1}}{n_{2}} \sin \theta_{\mathrm{i}}\right)^{2}}+n_{2} \cos \theta_{\mathrm{i}}}\right|^{2} 
\end{aligned}\right.
$$
Approximation: **Schlick’s approximation**
$$
R(\theta) = R_0 + (1-R_0)(1-\cos\theta)^5\\
R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2
$$

## Microfacet Material

### Microfacet Theory

View from far away: material and appearance; from nearby: geometry

Rough Suface:

- Macroscale: flat & rough
- Microscal: bumpy & specular

Individual elements of surface act like mirrors

- Known as Microfacets
- Each microfacet has its **own normal**  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223113238287.png" alt="image-20211223113238287" style="zoom:67%;" />

### Microfacet BRDF

Key: the distribution of **microfacets’ normals** => the **roughness** 

- concentrated distributed <==> glossy

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223113524613.png" alt="image-20211223113524613" style="zoom:80%;" />

- spread distributed <==> diffuse

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223113539391.png" alt="image-20211223113539391" style="zoom:80%;" />

Microfacets are mirrors

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223114726490.png" alt="image-20211223114726490" style="zoom:67%;" />

($\vb{F}$ - Fresnel term; $\vb{G}$ - shadowing-masking term (occlude with each other, usually occur when grazing angle light); $\vb{D}$ - distribution of normals)
$$
f(\vb{i,o}) = \frac{\vb{F(i,h)G(i,o,h)D(h)}}{4\vb{(n,i)(n,o)}}
$$

## Isotropic / Anisotropic Materials (BRDFs)

Key: **directionality** of underlying surface (following are the surface normals and the BRDF with fixed wi and varied wo)

- Isotropic: if different oriented azimuthal angles of incident and reflected rays give the same BRDFs, the material is isotropic

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223124457409.png" alt="image-20211223124457409" style="zoom:50%;" /> 

- Anisotropic (strongly directional)

  <img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211223124520003.png" alt="image-20211223124520003" style="zoom:50%;" /> 

  **Reflection depends on azimuthal angle** $\phi$: $f_r(\theta_i , \phi_i;\ \theta_r,\phi_r) \neq f_r(\theta_i , \theta_r, \phi_r - \phi_i)$ 

  Results from oriented microstructure of surface, e.g., brushed metal, nylon, velvet

## Properties of BRDFs

- Non-negativity: $f_r (\omega_i \rightarrow \omega_r)\ge 0$

- Linearity: $L_{r}\left(\mathrm{p}, \omega_{r}\right)=\int_{H^{2}} f_{r}\left(\mathrm{p}, \omega_{i} \rightarrow \omega_{r}\right) L_{i}\left(\mathrm{p}, \omega_{i}\right) \cos \theta_{i} \mathrm{~d} \omega_{i}$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223125730196.png" alt="image-20211223125730196" style="zoom:50%;" />

- Reciprocity principle: $f_r (\omega_r \rightarrow \omega_i) = f_r (\omega_i \rightarrow \omega_r)$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223125956218.png" alt="image-20211223125956218" style="zoom:50%;" />

- Energy conservation: (the convergence of path tracing)
  $$
  \forall \omega_{r} \int_{H^{2}} f_{r}\left(\omega_{i} \rightarrow \omega_{r}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \leq 1
  $$

- Isotropic vs. anisotropic

  - If isotropic: $f_r(\theta_i , \phi_i;\ \theta_r,\phi_r) \neq f_r(\theta_i , \theta_r, \phi_r - \phi_i)$
  - from reciprocity: $f_{r}\left(\theta_{i}, \theta_{r}, \phi_{r}-\phi_{i}\right)=f_{r}\left(\theta_{r}, \theta_{i}, \phi_{i}-\phi_{r}\right)=f_{r}\left(\theta_{i}, \theta_{r},\left|\phi_{r}-\phi_{i}\right|\right)$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223130142560.png" alt="image-20211223130142560" style="zoom:40%;" />

## Measuring BRDFs

### Image-Based BRDF Measurement

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223130441963.png" alt="image-20211223130441963" style="zoom:50%;" />

Instrument: gonioreflectometer

**General approach**: (curse of dimensionality)

for each outgoing dir wo
	move light to illuminate surface with a thin beam from wo
	for each incoming dir wi
		move sensor to be at dir wi from surface
		measure the incident radiance

**Improving efficiency**:

- Isotropic surfaces reduce dim from 4d -> 3d
- Reciprocity reduce # of measurements by half
- Clever optical systems

…

-> Important lib for BRDFs: MERL BRDF Database 



# Advanced Topics in Rendering (Lec. 18)

> Advanced light transport and materials

## Advaced Light Transport

- Unbiased light transport methods
  - Bidirectional path tracing (BDPT)
  - Metropolis light transport (MLT)
- Biased light transport methods
  - Photon mapping
  - Vertex connection and merging (VCM)
- Instant radiosity (VPL / many light methods)

**Biased vs. Unbiased Monte Carlo Estimators**

- An unbiased Monte Carlo doesn’t have any systematic error (No matter how many samples -> always expect the correct result)
- Otherwise, biased (special case: expected value converges to the correct value as infinite \#samples are used  \#samples are used - <u>consistent</u>)

### Unbiased Light Transport Methods

#### Bidirectional Path Tracing (BDPT)

A path connects the camera and the light

BDPT: Traces sub-paths from both the cam and the light; connects the end pts from both sub-paths

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223220006646.png" alt="image-20211223220006646" style="zoom:50%;" />

**Properties**:

- Suitable if the light transport is complex on the light’s side
- Difficult to implement & quite slow  

#### Metropolis Light Transport (MLT)

A Markov Chain Monte Carlo (MCMC) application

Jumping from the current samplke to the next with some PDF

Key idea: Locally perturb an existing path to get a new path

Good at <u>locally</u> exploring difficult light paths

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223221502771.png" alt="image-20211223221502771" style="zoom:50%;" />

**Properties**:

- Works great with difficult light paths
- Difficult to estimate the convergence rate; doesn’t guarantee equal convergence rate per pixel; usually produces “dirty” results
  => Usually not used to render animations

### Biased Light Transport Methods

#### Photon Mapping

A biased approach & A two-stage method -> Good at handling <u>Specular-Diffuse-Specular (SDS) paths</u> and generating <u>caustics</u>

**Approach (variations apply)**:

- Stage 1 - **Photon tracing**: Emitting photons from the light source, bouncing them around, then recording photons on diffuse surfaces  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211223222410114.png" alt="image-20211223222410114" style="zoom: 50%;" />

- Stage 2 - **Photon collection (final gathering)**: Shoot sub-paths from the camera, bouncing them around, until they hit diffuse surfaces  

**Calculation**: <u>Local density estimation</u>

Areas with more photons should be brighter. For each shading point find the nearest N photon. Take the surface area they over. (Density = Num / Area)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225105208322.png" alt="image-20211225105208322" style="zoom:50%;" />

Smaller $ N$ -> noisy; Larger $N$ -> blurry 

Biased method problem: local density est: $\mathrm{d}N/\mathrm{d}A $ != $\Delta N / \Delta A$; But in the sense of limit: more photons emitted -> the same $N$ photons cover a smaller $\Delta A$ -> $\Delta A \rightarrow \mathrm{d}A$ (Biased but consistent)

=> Bisaed == blurry; Consistent == not blurry with inf #samples

#### Vertex Connection and Merging (VCM)

A Combination of BDPT and Photon Mapping

Key: Not waste the sub-paths in BDPT if their end pt cannot be connected but can be merged; use photon mapping to handle the mergeing of nearby photons

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225112233746.png" alt="image-20211225112233746" style="zoom:50%;" />

### Instant Radiosity (IR)

= Many-light approaches

Key: Lit surfaces can be treated as light sources

**Approach**: 

- Shoot light sub-paths and assume the end point of each sub-path is a <u>Virtual Point Light</u> (VPL)
- Render the scene as usual using these VPLs  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225112704836.png" alt="image-20211225112704836" style="zoom: 67%;" />

**Properties**:

- Fast and usually gives good results on diffuse scenes  
- Spikes will emerge when VPLs are close to shading points; Cannot handle glossy materials  

## Advanced Appearance Modeling

- Non-surface models => many stuff seems like surface model but actually non-surface
  - Participating media
  - Hair / fur / fiber (BCSDF)
  - Granular material
- Surface models
  - Translucent material (BSSRDF)
  - Cloth
  - Detailed material (non-statistical BRDF)
- Procedural appearance

### Non-Surface Models

#### Participating Media

At any point as light travels through a participating medium, it can be (partially) <u>absorbed</u> and <u>scattered</u> (cloud / …)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225113516813.png" alt="image-20211225113516813" style="zoom:60%;" />

Use **Phase Function** (how to scatter) to describe the angular distribution of light scattering at any point $\vb{x}$ within participating media  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225113626278.png" alt="image-20211225113626278" style="zoom:67%;" />

**Rendering**:

- Randomly choose a direction to bounce
- Randomly choose a distance to go straight
- At each ‘shading point’, connect to the light  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225113834027.png" alt="image-20211225113834027" style="zoom:50%;" /> 

#### Hair Appearance

Light not on a surface but on a <u>thin cylinder</u> (Hightlight has 2 types: <u>color</u> and <u>colorless</u>)

##### Kajiya-Kay Model

diffuse + scatter (form a cone zone) => not real

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225114627302.png" alt="image-20211225114627302" style="zoom: 33%;" /> 

##### Marschner Model 

(widely used model) => very good results

some reflected R and some penantrated (refraction) T (-> TT / TRT /…)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225115036874.png" alt="image-20211225115036874" style="zoom:33%;" /> 

<u>Glass-like cylinder</u> (black -> absorb more; brown / bronde -> absorb less => color) + <u>3 types of light interactions</u> (R / TT / TRT)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225161647989.png" alt="image-20211225161647989" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225161805983.png" alt="image-20211225161805983" style="zoom:50%;" />

=> extremely high costs

#### Fur Appearance

cannot just simply apply human hairs => different in biological structures

**Common**: 3 layer structure

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225162152287.png" alt="image-20211225162152287" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225162211761.png" alt="image-20211225162211761" style="zoom:67%;" />

**Difference**: fur in animal has much bigger medulla (髓质) than human hair => more complex refraction inside (Need to simulate medulla for animal fur)

##### Double Cylinder Model

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225162523306.png" alt="image-20211225162523306" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225162549901.png" alt="image-20211225162549901" style="zoom:50%;" />

Add 2 scattered model => TT^s^ and TRT^s^ 

#### Granular Material

avoid explicit modeling => procedural definition

### Surface Models

#### Translucent Materials

Jellyfish / Jade / …

**Subsurface Scattering**  

light exiting at different points than it enters  

Actually violates a fundamental assumption of the BRDF (on the diffuse surface with BRDF, light reflects at the same point)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225163955908.png" alt="image-20211225163955908" style="zoom:67%;" />

**Scattering functions**: 

**BSSRDF**: generalization of BRDF; exitant radiance at one point due to incident differential irradiance at another point: $S(x_i , \omega_i , x_o, \omega_o)$ 

=> integrating over all points on the <u>surface</u> (area) and all <u>directions</u>  
$$
L\left(x_{o}, \omega_{o}\right)=\int_{A} \int_{H^{2}} S\left(x_{i}, \omega_{i}, x_{o}, \omega_{o}\right) L_{i}\left(x_{i}, \omega_{i}\right) \cos \theta_{i} \mathrm{~d} \omega_{i} \mathrm{~d} A
$$
**Dipole Approximation**

Approximate light diffusion by introducing <u>two point sources</u>   

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225164155969.png" alt="image-20211225164155969" style="zoom:50%;" />

#### Cloth

A collection of twisted fibers (different levels of twist: fibers -> ply -> yarn)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225164546984.png" alt="image-20211225164546984" style="zoom:67%;" />

##### Render as Surface

Given the weaving pattern, calculate the overall behavior (using <u>BRDF</u>)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225164655172.png" alt="image-20211225164655172" style="zoom: 80%;" />

=> limitation: anisotropic cloth

##### Render as Participating Media

Properties of individual fibers & their distribution -> scattering parameters  

=> Really high costs

##### Render as Actual Fibers

Render every fiber explicitly 

=> Similar to hair, extremely high costs

#### Detailed Appearance

The now renderers are not good => too perfect results (reality: straches / pores / …)

Recap: Microfacet BRDF

**Surface = <u>Specular</u> microfacets + <u>Statistical</u> normals**
$$
f(\vb{i,o}) = \frac{\vb{F(i,h)G(i,o,h)D(h)}}{4\vb{(n,i)(n,o)}}
$$
($\vb{D}$ - NDF: Normal Distribution Function, we have the left normal dist, but want the right - noises)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225165457893.png" alt="image-20211225165457893" style="zoom: 50%;" />

**Define Details**: required very high resolution normal maps 

=> too difficult for rendering: for bumpy specular surface -> hard to catch from both cam or light source

**Solution**: BRDF over a <u>pixel</u>

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225165818917.png" alt="image-20211225165818917" style="zoom:50%;" />

p-NDFs have sharp features 

=> ocean waves / scratch / …

**Recent Trend**: <u>Wave optics</u> (for too micro / short time duration) other than geometric optics

### Procedural Appearance

define details without textures => compute a noise function on the fly

- 3D noise -> internal structure
- hresholding (noise -> binary noise)



# Cameras, Lenses and Lightings (Lec. 19)

Imaging = Synthesis + Capture

The sensor records the <u>irradiance</u>

## Field of View (FOV)

### Effect of Focal Lenght on FOV

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225210551845.png" alt="image-20211225210551845" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225210604243.png" alt="image-20211225210604243" style="zoom:50%;" />

For a fixed sensor size, decreasing the focal length increases the field of view (assuming the sensor is fully used)
$$
\text{FOV} = 2\arctan\left(\frac{h}{2f}\right)
$$
The referred focal lengh of a lens used on a 35mm-format film (36 x 24 mm)

-> e.g., 17 mm is wide angle 104°; 50 mm is a “normal” lens 47°; 200 mm is a telephoto lens 12°  

### Effect of Sensor Size on FOV

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225211019286.png" alt="image-20211225211019286" style="zoom: 50%;" /> 

Maintain FOV on smaller sensor? => shorter focal length

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225211115882.png" alt="image-20211225211115882" style="zoom:67%;" />

## Exposure

**Exposure** $H$ = **Time** ($T$) $\times$ **Irradiance** ($E$) 

- Time: controlled by <u>shutter</u>
- Irradiance: <u>power</u> of light falling on a unit area of sensor; controlled by lens <u>aperture</u> and <u>focal lenth</u>

### Exposure Controls

- **Aperture size**

  - Change the <u>f-stop</u> (Exposure levels) by opening / closing the aperture (iris control)

    `FN` or `F/N`: the inverse-diameter of a round aperture

    The f-number of a lens is defined as <u>the focal length divided by the diameter of the aperture</u>  

- **Shutter speed **(causes motion blur in slower speed / rolling shutter)

  - Change the <u>duration</u> the sensor pixels integrate light

- **ISO gain** (Trade sensitivity of grain / noise)

  - Change the <u>amplification</u> (analog / digital) between the sensor values and digital image values

**Some pairs**

| F-stop  | 1.4   | 2.0   | 2.8   | 4.0  | 5.6  | 8.0  | 11.0 | 16.0 | 22.0 | 32.0 |
| ------- | ----- | ----- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Shutter | 1/500 | 1/250 | 1/125 | 1/60 | 1/30 | 1/15 | 1/8  | 1/4  | 1/2  | 1    |

### Fast and Slow Photography  

- High-Speed Photography   

  Normal exposure = extremely fast shutter speed x (large aperture and/or high ISO)  

- Long-Exposure Photography  

## Thin Lens Approximation  

Real Lens Elements Are Not Ideal – <u>Aberrations</u>  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225220453824.png" alt="image-20211225220453824" style="zoom: 50%;" />

### Ideal Thin Lens – Focal Point  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225220523732.png" alt="image-20211225220523732" style="zoom:50%;" />

- All parallel rays entering a lens pass through its focal point
- All rays through a focal point will be in parallel after passing the lens
- Local length can be arbitrarily changed (actually in camera lens -> yes)

### The Thin Lens Equation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225220644027.png" alt="image-20211225220644027" style="zoom: 33%;" />

**Gaussian Thin Lens Equation**
$$
\frac{1}{f} = \frac{1}{z_i} + \frac{1}{z_o}
$$

### Defocus Blur

#### Computing Circle of Confusion (CoC) Size  

If not at the focal plane -> not projected at the sensor plane -> blurry 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225220925821.png" alt="image-20211225220925821" style="zoom:50%;" />

Circle of confusion is <u>proportional to the size of the aperture</u>  
$$
\frac{C}{A} = \frac{d'} {z_i} = \frac{|z_s - z_i|}{z_i}
$$

### Ray Tracing for Defocus Blur (Thin Lenses)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225223534104.png" alt="image-20211225223534104" style="zoom:50%;" />

(One possible) **Setup**:

- Choose sensor size, lens focal length and aperture size
- Choose depth of subject of interest $z_o$
- Calculate corresponding depth of sensor $z_i$ from thin lens equation  

**Rendering**:

- For each pixel $x’$ on the sensor (actually, film)
- Sample random points $x’’$ on lens plane
- You know the ray passing through the lens will hit $x’’’$ (because $x’’’$ is in focus, consider virtual ray ($x’$, center of the lens))
- Estimate radiance on ray $x’’ \rightarrow x’’’  $

## Depth of Field

### Circle of Confusion for Depth of Field  

Depth range in a scene where the corresponding CoC is considered small enough  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225223930501.png" alt="image-20211225223930501" style="zoom:50%;" />

### Depth of Field

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211225224005805.png" alt="image-20211225224005805" style="zoom:67%;" />
$$
\mathrm{DOF} = D_F - D_N\\
D_{F}=\frac{D_{S} f^{2}}{f^{2}-N C\left(D_{S}-f\right)} \quad D_{N}=\frac{D_{S} f^{2}}{f^{2}+N C\left(D_{S}-f\right)}
$$

## Light Field / Lumigraph







# Color and Perception (Lec. 20)



















































































# Animation (Lec. 21-22) 

