# GAMES103 - Intro to Physics-Based Animation

[![](https://img.shields.io/badge/Main%20Page-%20%20-blueviolet)](https://nikucyan.github.io/) [![](https://img.shields.io/badge/Repo-%20%20-blue)](https://github.com/Nikucyan/Notes_of_Graphics/tree/main/GAMES103) [![](https://img.shields.io/badge/HW-%20%20Codes-yellow)](https://github.com/Nikucyan/Notes_of_Graphics/tree/main/GAMES103/Homework_Assignments)

(Based on Unity, C#)

> Huamin Wang (games103@style3D.com)	[Video](https://www.bilibili.com/video/BV12Q4y1S73g); [Lecture site](http://games-cn.org/games103/)



# Lecture 1 Introduction

## Graphics

### Geometry

- **Mesh**: Triangle mesh is the foundation of graphics

  Vertices (nodes) + Elements (triangles, polygons, tetrahedra…)

  ![image-20211102113748489](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102113748489.png)

- **Point Cloud**: simple, can be raw data from surface scan. 

  But problems in mesh reconstruction, (re)-sampling, neighborhood search …

- **Grid**: often acquired from volumetric scan, e.g., CT

  Problems in memory cost (octree?), volumetric rendering

### Rendering 

- Photorealistic Rendering (Ray Tracing)
- Non-Photorealistic Rendering

**Material Scan**

- Body Scan by a Light Stage

### Animations 

- Character Animation
- Physics-Based Animation

## Physics-Based Animation Topics

- **Rigid Bodies** [<u>contact</u> / fracture] - <u>Mesh</u> / *Particle in fracture (to avoid remeshing)
- **Cloth and Hair** [<u>clothes</u> / hair] - <u>Mesh</u> / *Grid (to simplify contacts)
- **Soft Bodies** [<u>elastic</u> / plastic] - Mesh
- **Fluids** [smoke / drops and waves / splashes] - Mesh in drops and waves (RT) / <u>Particle</u> (RT) in smoke and splashes / <u>Grid</u> (universal)

**Hybrid Methods and Coupling** - problems



# Lecture 2 Math Background

> **Vector, Matrix and Tensor Calculus** 

## Vectors

### Definitions

A geometric entity endowed with magnitude and direction
$$
\vb{p} = \begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix} \in \R ^3 \ ; \quad  \vb{o} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}
$$

- Right-Hand System (OpenGL, Research, …) 
- Left-Hand System (Unity, DirectX, …)  => screen space

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109004814489.png" alt="image-20211109004814489" style="zoom:63%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109004854038.png" alt="image-20211109004854038" style="zoom: 67%;" />

Can also be stacked up to form a high-dim vector -> describe the state of an obj (not a geometric vector but a stacked vector)

### Arithematic

#### **Addition and Substraction**

 (commutative)
$$
\vb{p} \pm \vb{q} = \begin{bmatrix} p_x \pm q_x \\ p_y \pm q_y \\ p_z \pm q_z\end{bmatrix}\ ;\quad 
\vb{p} + \vb{q} = \vb{q} + \vb{p}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109005357876.png" alt="image-20211109005357876" style="zoom:50%;" />

**Example**: $\vb{p}(t) = \vb{p} + t\vb{v}$ to represent the movement of a particle. Segment: $0 < t<1$ ; Ray: $0<t$; Line: $t\in \R$  ($t$ is an interpolant)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109005650429.png" alt="image-20211109005650429" style="zoom:60%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109005758487.png" alt="image-20211109005758487" style="zoom:50%;" />

#### Vector **Norm**

Magnitude of a vector (length)

- **1-norm**: $||\vb{p}||_1=|p_x|+|p_y |+|p_z|$
- **Euclidean (2) norm **(default): $||\vb{p}||_2 = (p_x^2 + p_y^2 + p_z^2) ^{1/2}$
- **p-norm**: $||\vb{p}||_p = (p_x^p + p_y^p + p_z^p) ^{1/p}$
- **Inifinity norm** (Maximum): $||\vb{p}||_\infty=\max(|p_x |,|p_x |,|p_x |)$

**Usage**: 

- **Distance** between $\vb{p}$ and $\vb{q}$: $||\vb{q}-\vb{p}||$
- **Unit Vector**: $||\vb{p}|| = 1$
- **Normalization**: $\bar {\vb{p}} = \vb{p} / ||\vb{p}||$ as $||\bar {\vb{p}}|| = ||\vb{p}|| / ||\vb{p}|| = 1$

#### **Dot Product** 

(inner product)

$$
<\vb{p},\vb{q}> \equiv \vb{p}\cdot \vb{q} = p_xq_x + p_yq_y + p_zq_z = \vb{p}^{\mathrm{T}}\vb{q} = ||\vb{p}||\ ||\vb{q}|| \cos\theta
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109104509904.png" alt="image-20211109104509904" style="zoom: 67%;" />

**Properties**:

- $\vb{p}\cdot \vb{q} = \vb{q}\cdot \vb{p} $
- $\vb{p}\cdot \vb{(q+r)} = \vb{p}\cdot \vb{q}  + \vb{p}\cdot \vb{r} $
- $\vb{p}\cdot \vb{p} = ||\vb{p}||^2_2$ , an alternative way to write norm
- If $\vb{p}\cdot \vb{q}  = 0$ and $\vb{p}, \vb{q} \neq 0$, then $\cos \theta = 0$ => **orthogonal**

**Example**: Particle-Line Projection

By def: $s = ||\vb{q} - \vb{o}||\ \cos \theta$ => $\vb{s} = \vb{o} - s\bar{\vb{v}}$
$$
\begin{aligned}
s &= ||\vb{q} - \vb{o}||\ \cos \theta \\
&= ||\vb{q} - \vb{o}||\ ||\vb{v}|| \cos \theta\ / ||\vb{v}|| \\
&= (\vb{q} - \vb{o}) ^{\mathrm{T}} \vb{v} / ||\vb{v}|| \\
&= (\vb{q} - \vb{o}) ^{\mathrm{T}} \bar{\vb{v}} \quad (\mathrm{normalized})
\end{aligned}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109104953250.png" alt="image-20211109104953250" style="zoom:50%;" />

**Example**: Plane Representation

Check the relationship between point p and plane ($s$ - the signed distance to the plane -> collision check / …; $\vb{n}$ - normal)
$$
s = (\vb{p} - \vb{c})^{\mathrm{T}}\vb{n} \ 
\left\{
\begin{aligned}
>0\quad & \text{Above the plane}\\
=0\quad & \text{On the plane}\\
<0\quad & \text{Below the plane}
\end{aligned}
\right.
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109105816092.png" alt="image-20211109105816092" style="zoom:50%;" />

**Example**: Particle-Sphere Collision
$$
\begin{aligned}
||\vb{p}(t) - \vb{c}||^2 &= r^2\\
(\vb{p}-\vb{c}+t\vb{v})\cdot (\vb{p}-\vb{c}+t\vb{v})&=𝑟^2\\
(\vb{v}\cdot \vb{v})t^2 + 2(\vb{p} - \vb{c})\cdot \vb{v}t + (\vb{p} - \vb{c})\cdot (\vb{p} - \vb{c}) - r^2 &= 0
\end{aligned}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109105955443.png" alt="image-20211109105955443" style="zoom:50%;" />

$t$ is the root -> No root (no collision) / One root (tangentially) / Two roots (the first point, smaller $t$)

#### Cross Product

$$
\vb{r } = \vb{p}\times\vb{q} = \begin{bmatrix}p_yq_z - p_zq_y \\ p_z q_x - p_xq_z \\ p_xq_y - p_yq_x \end{bmatrix}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109110925156.png" alt="image-20211109110925156" style="zoom:50%;" />

**Properties**

- $\vb{r} \cdot \vb{p} = 0$; $\vb{r} \cdot \vb{q} = 0$; $||\vb{r}|| = ||\vb{p}|| \ ||\vb{q}|| \ \sin\theta$
- $\vb{p}\times\vb{q} = - \vb{q}\times\vb{p}$
- $\vb{p} \times(\vb{q}+ \vb{r}) = \vb{p}\times\vb{q}  + \vb{p}\times\vb{r} $
- If $\vb{p}\times\vb{q} = \vb{0}$ and $\vb{p}, \ \vb{q} \neq 0$, then $\sin\theta = 0$, $\vb{p}$ & $\vb{q}$ are **parallel** (direction can be opposite)

**Example**: Trinagle Normal and Area 

- Edge vectors: $\vb{x}_{10} = \vb{x}_1 - \vb{x}_0$ & $\vb{x}_{20} = \vb{x}_2 - \vb{x}_0$
- Normal: $\vb{n} = (\vb{x}_{10} \times\vb{x}_{20}) / ||\vb{x}_{10} \times\vb{x}_{20}||$ (dep on the topological order)
- Area: $A = ||\vb{x}_{10}|| h /2 = ||\vb{x}_{10} \times\vb{x}_{20}||/2$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109112748339.png" alt="image-20211109112748339" style="zoom:50%;" />

**Example**: Trianle Inside / Outside Test (Co-plane)

- If inside $\vb{x}_0 \vb{x_1}$: $(\vb{x}_0 -\vb{p})\times (\vb{x}_1 - \vb{p})\cdot \vb{n} > 0$ (Same normal as the main triangle)
- If outside $\vb{x}_0\vb{x}_1$: $(\vb{x}_0 -\vb{p})\times (\vb{x}_1 - \vb{p})\cdot \vb{n} < 0$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109113822850.png" alt="image-20211109113822850" style="zoom:50%;" />

**Example**: Barycentric Coordinates
$$
\frac{1}{2}\left(\mathbf{x}_{0}-\mathbf{p}\right) \times\left(\mathbf{x}_{1}-\mathbf{p}\right) \cdot \mathbf{n}=\left\{\begin{array}{r}
\frac{1}{2}\left\|\left(\mathbf{x}_{0}-\mathbf{p}\right) \times\left(\mathbf{x}_{1}-\mathbf{p}\right)\right\| &\quad \text{inside}\\
-\frac{1}{2}\left\|\left(\mathbf{x}_{0}-\mathbf{p}\right) \times\left(\mathbf{x}_{1}-\mathbf{p}\right)\right\| &\quad\text{outside}
\end{array}\right.
$$
Signed Areas:
$$
\begin{aligned}
&A_{2}=\frac{1}{2}\left(\mathbf{x}_{0}-\mathbf{p}\right) \times\left(\mathbf{x}_{1}-\mathbf{p}\right) \cdot \mathbf{n} \\
&A_{0}=\frac{1}{2}\left(\mathbf{x}_{1}-\mathbf{p}\right) \times\left(\mathbf{x}_{2}-\mathbf{p}\right) \cdot \mathbf{n} \\
&A_{1}=\frac{1}{2}\left(\mathbf{x}_{2}-\mathbf{p}\right) \times\left(\mathbf{x}_{0}-\mathbf{p}\right) \cdot \mathbf{n} \\
&A = A_0 + A_1 + A_2
\end{aligned}
$$
Barycentric weights of $\vb{p}$: $b_0 = A_0/A$, $b_1 = A_1/A$, $b_2 = A_2/A$ ($b_0 + b_1 +b_2 = 1$)

Barycentric Interpolation: $\vb{p} = b_0\vb{x}_0 + b_1\vb{x}_1 + b_2 \vb{x}_2$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109114230534.png" alt="image-20211109114230534" style="zoom:50%;" />

​	=> **Gourand Shading**: Using barycentric interpolation (no longer popular)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109114950142.png" alt="image-20211109114950142" style="zoom:50%;" />

**Example**: Tetrahedral Volume

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109115747395.png" alt="image-20211109115747395" style="zoom:50%;" />

- Edge vectors: $\vb{x}_{10} = \vb{x}_1 - \vb{x}_0$ & $\vb{x}_{20} = \vb{x}_2 - \vb{x}_0$ & $\vb{x}_{20} = \vb{x}_2 - \vb{x}_0$

- Base triangle area: $A = \frac{1}{2} ||\vb{x}_{10} - \vb{x} _ {20}||$

- Height: $h = \vb{x}_{30}\cdot \vb{n} = \vb{x}_{30} \cdot \frac{\vb{x}_{10} - \vb{x} _ {20}}{||\vb{x}_{10} - \vb{x} _ {20}||}$

- Volume: (signed)
  $$
  V= \frac{1}{3}hA = \frac{1}{6} \vb{x}_{30} \cdot \vb{x}_{10} \times \vb{x}_{20} = \frac{1}{6}\begin{bmatrix}\vb{x}_1 & \vb{x}_2 & \vb{x}_3 & \vb{x}_0 \\ 1 & 1 & 1& 1\end{bmatrix}
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109115820905.png" alt="image-20211109115820905" style="zoom: 50%;" />

**Example**: Barycentric Weight in Tetrahedra

$\vb{p}$ splits the tetrahedron into 4 tetrahedra: $V_0 = \mathrm{Vol}(\vb{x}_3, \vb{x}_2, \vb{x}_1, \vb{p})$ , ….

$\vb{p}$ inside: if and only if $V_0, V_1,V_2,V_3 >0$

Barycentric weights: $b_n = V_n /V$,  $\vb{p} = b_0\vb{x}_0 +  b_1\vb{x}_1 +  b_2\vb{x}_2 +  b_3\vb{x}_3$

**Example**: Particle-Triangle Intersection

- First find $t$ when particle hits the plane: $(\vb{p}(t) - \vb{x}_0)\cdot \vb{x}_{10}\times\vb{x}_{20} = 0$, where $\vb{p}(t) = \vb{p} + t\vb{v}$
  $$
  t=\frac{\left(\mathbf{p}-\mathbf{x}_{0}\right) \cdot \mathbf{x}_{10} \times \mathbf{x}_{20}}{\mathbf{v} \cdot \mathbf{x}_{10} \times \mathbf{x}_{20}}
  $$

- Check $\vb{p}(t)$ inside or outside

## Matrices

### Definitions

$$
\mathbf{A}=\left[\begin{array}{lll}
a_{00} & a_{01} & a_{02} \\
a_{10} & a_{11} & a_{12} \\
a_{20} & a_{21} & a_{22}
\end{array}\right]=\left[\begin{array}{lll}
\mathbf{a}_{0} & \mathbf{a}_{1} & \mathbf{a}_{2}
\end{array}\right] \in \R^{3 \times 3}
$$

- Transpose / Diagonal / Identity / Symmetric:
  $$
  \mathbf{A}^{\mathrm{T}}=\left[\begin{array}{lll}
  a_{00} & a_{01} & a_{02} \\
  a_{10} & a_{11} & a_{12} \\
  a_{20} & a_{21} & a_{22}
  \end{array}\right]
  \ ;\quad \quad
  \begin{bmatrix}
  a_{00} & & \\
   & a_{11} & \\
   & & a_{22}
  \end{bmatrix}\ ;
  \quad\quad
  \vb{I} = \begin{bmatrix}
  1 & & \\
   & 1 & \\
   & & 1\end{bmatrix}\ ;
  \quad\quad
  \vb{A}^{\mathrm{T}} = \vb{A}
  $$

### Orthogonality

An orthogonal matrix is a matrix made of orthogonal unit vectors.
$$
\vb{A} = \begin{bmatrix} \vb{a}_0 & \vb{a}_1 & \vb{a}_2\end{bmatrix}\ ,\ \text{such that }\ 
\vb{a}_i^{\mathrm{T}} \vb{a}_j = \left\{ 
\begin{aligned}
1, \quad& \text{if}\ i = j\\
0, \quad& \text{if}\ i\neq j
\end{aligned}
\right.
$$

$$
\mathbf{A}^{\mathrm{T}} \mathbf{A}=\left[\begin{array}{c}
\mathbf{a}_{0}^{\mathrm{T}} \\
\mathbf{a}_{1}^{\mathrm{T}} \\
\mathbf{a}_{2}^{\mathrm{T}}
\end{array}\right]\left[\begin{array}{lll}
\mathbf{a}_{0} & \mathbf{a}_{1} & \mathbf{a}_{2}
\end{array}\right]=\left[\begin{array}{ccc}
\mathbf{a}_{0}^{\mathrm{T}} \mathbf{a}_{0} & \mathbf{a}_{0}^{\mathrm{T}} \mathbf{a}_{1} & \mathbf{a}_{0}^{\mathrm{T}} \mathbf{a}_{2} \\
\mathbf{a}_{1}^{\mathrm{T}} \mathbf{a}_{0} & \mathbf{a}_{1}^{\mathrm{T}} \mathbf{a}_{1} & \mathbf{a}_{1}^{\mathrm{T}} \mathbf{a}_{2} \\
\mathbf{a}_{2}^{\mathrm{T}} \mathbf{a}_{0} & \mathbf{a}_{2}^{\mathrm{T}} \mathbf{a}_{1} & \mathbf{a}_{2}^{\mathrm{T}} \mathbf{a}_{2}
\end{array}\right]=\mathbf{I}\ ; \quad
\vb{A}^{\mathrm{T}} = \vb{A}^{-1}
$$

### Matrix Transformation

- **Rotation** can be represented by an **orthogonal** matrix 

  (can represent local -> world)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109164837796.png" alt="image-20211109164837796" style="zoom:50%;" />

  (Considering of local coord. vect.)
  $$
  \left.
  \begin{aligned}
  &\vb{A} \vb{x} = \vb{u}\\
  &\vb{A} \vb{y} = \vb{v}\\
  &\vb{A} \vb{z} = \vb{w}
  \end{aligned}\right\}
  \Rightarrow \vb{A} = 
  \begin{bmatrix} \vb{u}&\vb{v}&\vb{w}\end{bmatrix}
  $$

- **Scaling** can be represented by a **diagonal** matrix

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109165018049.png" alt="image-20211109165018049" style="zoom:50%;" />

  (Consisting of scaling factors)
  $$
  \vb{D} = \begin{bmatrix}d_x & & \\ & d_y & \\ & & d_z \end{bmatrix}
  $$

- **Singular Value Decomposition**

  A matrix can be decomposed $\vb{A} = \vb{UDV}^{\mathrm{T}}$  ($\vb{D}$ - Diagonal, $\vb{U}$ & $\vb{V}$ - Orthogonal)

  Rotation ($\vb{V}^{\mathrm{T}}$) -> Scaling ($\vb{D}$) -> Rotation (All can be decomposed as rotaion and scaling (even in 3D))

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109165409060.png" alt="image-20211109165409060" style="zoom:70%;" />

- **Eigenvalue Decomposition**

  Only consider **symmetric** matrices: $\vb{A} = \vb{UDU}^{-1} = \vb{UDU}^{\mathrm{T}}$ 

  Also can be defined: Let $\vb{U} = \begin{bmatrix}\cdots & \vb{u}_i & \cdots\end{bmatrix}$, $\vb{u}_i$ is the eigenvector of $d_i$
  $$
  \mathbf{A} \mathbf{u}_{i}=\mathbf{U D U} ^{\mathrm{T}} \mathbf{u}_{i}=\mathbf{U D}\left[\begin{array}{c}
  \vdots \\
  0 \\
  1 \\
  0 \\
  \vdots
  \end{array}\right]=\mathbf{U}\left[\begin{array}{c}
  \vdots \\
  0 \\
  d_{i} \\
  0 \\
  \vdots
  \end{array}\right]=d_{i}\mathbf{u}_{i}
  $$
  For asymmetric matrices -> eigenvalue can be imaginary nums

- **Symmetric Positive Definiteness (s.p.d.)**

  -> Linear system

  For a s.p.d.,  if only if all its eigenvalues are positive

  - $\vb{A}$ is s.p.d. if and only if: $\vb{v}^{\mathrm{T}}\vb{Av} >0$, for any $\vb{v} \ne 0$
  - $\vb{A}$ is symmetric semi-definite if and only if: $\vb{v}^{\mathrm{T}}\vb{Av} \ge 0$, for any $\vb{v} \ne 0$

  Meaning: 

  - for $d>0$ => $\vb{v}^{\mathrm{T}}d\vb{v} > 0$ for any $\vb{v} \ne0$;

  - for $d_0, d_1, ... >0$  (for any $\vb{v} \ne0$)
    $$
    \Rightarrow\vb{v}^{\mathrm{T}}\vb{Dv} = \vb{v}^{\mathrm{T}}
    \begin{bmatrix} \ddots & & \\ & d_i & \\ & & \ddots \end{bmatrix}
    \vb{v} = 
    \sum d_i v_i^2
    > 0
    $$

  - for $d_0, d_1, ... >0$ with a $\vb{U}$ orthogonal (for any $\vb{v} \ne0$)
    $$
    \Rightarrow \mathbf{v}^{\mathrm{T}}\left(\mathbf{U D U}^{\mathrm{T}}\right) \mathbf{v}=\mathbf{v}^{\mathrm{T}} \mathbf{U U}^{\mathrm{T}}\left(\mathbf{U D U}^{\mathrm{T}}\right) \mathbf{U U}^{\mathrm{T}} \mathbf{v}
    = (\vb{U}^{\mathrm{T}}\vb{v})^{\mathrm{T}}(\vb{D}) (\vb{U}^{\mathrm{T}}\vb{v}) > 0
    $$

  In practice, a **diagonally domiant** matrix is p.d. ($a_{ii} > \sum_{i\ne j}|a_{ij}|$ for all $i$)

  - **Properties**:
  - $\vb{A}$ is s.p.d, then $\vb{B} = \begin{bmatrix}\vb{A}&\vb{-A}\\ \vb{-A}& \vb{A}\end{bmatrix}$ is symmetric semi-definite.

### Linear Solver

$\vb{Ax} = \vb{b}$ 	($\vb{A}$ - Square matrix; $\vb{x}$ - Unknown to be found; $\vb{b}$ - Boundary vector)

It’s expensive to compute $\vb{A}^{-1}$, especially if $\vb{A}$ is large and sparse (Cannot use $\vb{x} = \vb{A}^{-1}\vb{b}$)

#### Direct Linear Solver

**LU factorization** (Alt: Choleskey, LDL^T^, etc.)
$$
\vb{A} = \vb{LU} = \underbrace{\left[\begin{array}{ccc}
l_{00} & & \\
l_{10} & l_{11} & \\
\vdots & \cdots & \ddots
\end{array}\right]}_{\vb{L}}
\underbrace{\left[\begin{array}{ccc}
\ddots & \ldots & \vdots \\
& u_{n-1, n-1} & u_{n-1, n} \\
& & u_{n, n}
\end{array}\right]}_{\vb{U}}
$$
First solve: (up -> down)
$$
\vb{Ly = b} \Rightarrow \left[\begin{array}{ccc}
l_{00} & & \\
l_{10} & l_{11} & \\
\vdots & \cdots & \ddots
\end{array}\right]\left[\begin{array}{c}
y_{0} \\
y_{1} \\
\vdots
\end{array}\right]=\left[\begin{array}{c}
b_{0} \\
b_{1} \\
\vdots
\end{array}\right]\quad\Rightarrow
\left\{ \begin{align}& y_0 = b_0 / l_{00} \\ &y_1 = (b_1 - l_{10}y_0) / l_{11} \\ &...\end{align}\right.
$$
Then solve: (down -> up)
$$
\vb{Ux = y} \Rightarrow \left[\begin{array}{ccc}
\ddots & \ldots & \vdots \\
& u_{n-1, n-1} & u_{n-1, n} \\
& & u_{n, n}
\end{array}\right]\left[\begin{array}{c}
\vdots \\
x_{n-1} \\
x_{n}
\end{array}\right]=\left[\begin{array}{c}
\vdots \\
y_{n-1} \\
y_{n}
\end{array}\right]
\quad
\Rightarrow
\left\{ 
\begin{aligned}
&x_{n}=y_{n} / u_{n, n} \\
&x_{n-1}=\left(y_{n-1}-u_{n-1, n} x_{n}\right) / u_{n-1, n-1} \\
&\ldots
\end{aligned}
\right.
$$
**Properties**: 

- When $\vb{A}$ is sparse, $\vb{L}$ & $\vb{U}$ are not that sparse, dep on the **permutation** (modify the order) -> MATLAB (LUP)
- 2 steps: factorization & solving. if want more systems with the same $\vb{A}$, factorization could be done once (Save costs)
- Hard to **parallelized** (Intel MKL PARDISO)

#### Iterative Linear Solver

$$
\vb{x}^{[k+1]} = \vb{x}^{[k]} + \alpha \vb{M}^{-1} (\vb{b - Ax}^{[k]})
$$

($\alpha$ - relaxation; $\vb{M}$ - Iterative matrix; $\vb{b - Ax}^{[k]}$ - Residual error (for perfect solution -> = 0))

Converge property: ($\vb{b-Ax}^{[0]} = \text{const}$ at first)
$$
\begin{aligned}
\mathbf{b}-\mathbf{A} \mathbf{x}^{[k+1]}&=\mathbf{b}-\mathbf{A} \mathbf{x}^{[k]}-\alpha \mathbf{A} \mathbf{M}^{-1}\left(\mathbf{b}-\mathbf{A} \mathbf{x}^{[k]}\right)\\
&={\left(\mathbf{I}-\alpha \mathbf{A} \mathbf{M}^{-1}\right)\left(\mathbf{b}-\mathbf{A} \mathbf{x}^{[k]}\right)}=\left(\mathbf{I}-\alpha \mathbf{A} \mathbf{M}^{-1}\right)^{k+1}\left(\mathbf{b}-\mathbf{A} \mathbf{x}^{[0]}\right)
\end{aligned}
$$
So: ($\rho\left(\mathbf{I}-\alpha \mathbf{A M}^{-1}\right)$ - Spectral radius (largest absolute value of eigenvalues))
$$
\mathbf{b}-\mathbf{A} \mathbf{x}^{[k+1]} \rightarrow \mathbf{0},\text{ if }\rho\left(\mathbf{I}-\alpha \mathbf{A M}^{-1}\right)<1
$$
For $\vb{M}$, must be easy to solve, e.g. $\vb{M} = \mathrm{diag}(\vb{A})$ (**Jacobi**), or $\vb{M} = \text{lower}(\vb{A})$ (**Gauss-Seidel**)

Accelerate converge methods: Chebyshev, Conjugate Gradient, …

**Properties**:

- Simple; Fast for inexact sol; Parallelable
- Convergence condition (not converge for every matrix); Slow for exact solutions

## Tensor Calculus

### Basic Concepts

- 1st-Order Derivatives

  If $f(\vb{x})\in \R$, then 
  $$
  \mathrm{d}f  =\frac{\partial f}{\partial x} \mathrm{d} x+\frac{\partial f}{\partial y} \mathrm{d} y+\frac{\partial f}{\partial z} \mathrm{d} z=\begin{bmatrix}
  \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial z}
  \end{bmatrix}\left[\begin{array}{l}
  \mathrm{d} x \\
  \mathrm{d} y \\
  \mathrm{d} z
  \end{array}\right];\quad
  \frac{\partial f}{\partial\vb{x}} = \left[\begin{array}{lll}
  \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial z}
  \end{array}\right]
  \ \ \text{or}\ \
  \grad f(\vb{x}) = 
  \begin{bmatrix}\frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \\ \frac{\partial f}{\partial z}
  \end{bmatrix}
  $$
  If $\vb{f(x)} = \begin{bmatrix}f(\vb{x})\\ g(\vb{x}) \\ h(\vb{x})\end{bmatrix}\in \R^3$, then
  $$
  \text{Jacobian:}\ \mathbf{J}(\mathbf{x})=\frac{\partial \mathbf{f}}{\partial \mathbf{x}}=\left[\begin{array}{lll}
  \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} & \frac{\partial f}{\partial z} \\
  \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} & \frac{\partial g}{\partial z} \\
  \frac{\partial h}{\partial x} & \frac{\partial h}{\partial y} & \frac{\partial h}{\partial z}
  \end{array}\right];\quad 
  \text{Divergence:}\ \div\ \mathbf{f}=\frac{\partial f}{\partial x}+\frac{\partial g}{\partial y}+\frac{\partial h}{\partial z};\quad
  \text{Curl: }\curl\ \mathbf{f}=\left[\begin{array}{l}
  \frac{\partial h}{\partial y}-\frac{\partial g}{\partial z} \\
  \frac{\partial f}{\partial z}-\frac{\partial h}{\partial x} \\
  \frac{\partial g}{\partial x}-\frac{\partial f}{\partial y}
  \end{array}\right]
  $$

- 2nd-Order Derivatives

  If $f(\vb{x})\in \R$, then (Hessian is symmetric, tangent stiffness)
  $$
  \text{Hessian:}\ \mathbf{H}=\mathbf{J}(\nabla f(\mathbf{x}))=\left[\begin{array}{ccc}
  \frac{\partial^{2} f}{\partial x^{2}} & \frac{\partial^{2} f}{\partial x \partial y} & \frac{\partial^{2} f}{\partial x \partial z} \\
  \frac{\partial^{2} f}{\partial x \partial y} & \frac{\partial^{2} f}{\partial y^{2}} & \frac{\partial^{2} f}{\partial y \partial z} \\
  \frac{\partial^{2} f}{\partial x \partial z} & \frac{\partial^{2} f}{\partial y \partial z} & \frac{\partial^{2} f}{\partial z^{2}}
  \end{array}\right]
  ;\quad
  \text{Laplacian: }
  
  \nabla \cdot \nabla f(\mathbf{x})=\nabla^{2} f(\mathbf{x})= 
  \Delta f(\mathbf{x})=\frac{\partial^{2} f}{\partial x^{2}}+\frac{\partial^{2} f}{\partial y^{2}}+\frac{\partial^{2} f}{\partial z^{2}}
  $$

- Taylor Expansion

  If $f({x})\in \R$, then
  $$
  f(x)=f\left(x_{0}\right)+\frac{\partial f\left(x_{0}\right)}{\partial x}\left(x-x_{0}\right)+\frac{1}{2} \frac{\partial f^{2}\left(x_{0}\right)}{\partial x^{2}}\left(x-x_{0}\right)^{2}+\cdots
  $$
  If (vector func) $f(\vb{x})\in \R$, then
  $$
  \begin{aligned}
  f(\mathbf{x}) &=f\left(\mathbf{x}_{0}\right)+\frac{\partial f\left(\mathbf{x}_{0}\right)}{\partial \mathbf{x}}\left(\mathbf{x}-\mathbf{x}_{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x}_{0}\right)^{\mathrm{T}} \frac{\partial f^{2}\left(\mathbf{x}_{0}\right)}{\partial \mathbf{x}^{2}}\left(\mathbf{x}-\mathbf{x}_{0}\right)+\cdots \\
  &=f\left(\mathbf{x}_{0}\right)+\nabla f\left(\mathbf{x}_{0}\right) \cdot\left(\mathbf{x}-\mathbf{x}_{0}\right)+\frac{1}{2}\left(\mathbf{x}-\mathbf{x}_{0}\right)^{\mathrm{T}} \mathbf{H}\left(\mathbf{x}-\mathbf{x}_{0}\right)+\cdots
  \end{aligned}
  $$
  For $\vb{H}$ is s.p.d., second order derivative > 0 => interesting properties (to be discussed)



# Lecture 3 Rigid Body Dynamics

> (Single rigid body: dynamics / rotation / …)

Rigid bodies: Assume no deformations

The goal of simulation is to update the state var. $\vb{s}^{[k]}$ ove

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211117163642294.png" alt="image-20211117163642294" style="zoom:67%;" />

## Translation Motion

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211117215600450.png" alt="image-20211117215600450" style="zoom:67%;" />

For translation motion, the state variable contains the position $\vb{x}$ and the velocity $\vb{v}$ ($M$ - Mass; Force can be the function of pos, vel, t, …)  -> Solve integral
$$
\left\{\begin{aligned}
&\mathbf{v}\left(t^{[1]}\right)=\mathbf{v}\left(t^{[0]}\right)+M^{-1} \int_{t^{[0]}}^{t^{[1]}} \mathbf{f}(\mathbf{x}(t), \mathbf{v}(t), t)\ \mathrm d t \\
&\mathbf{x}\left(t^{[1]}\right)=\mathbf{x}\left(t^{[0]}\right)+\int_{t^{[0]}}^{t^{[1]}} \mathbf{v}(t)  \  \mathrm d  t
\end{aligned}\right.
$$

### Integration Methods Explained

By def, the integral of $\vb{x}(t) = \int \vb{v}(t)\ \mathrm d t$ is the area.

- **Explicit Euler** (1st-order accurate) sets the height at $t^{[0]}$ 
  $$
  \int_{t^{[0]}}^{t^{[1]}} \vb{v}(t)\ \mathrm{d}t \approx \Delta t \ \vb{v}(t^{[0]})
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211117220220499.png" alt="image-20211117220220499" style="zoom:50%;" />

  Use Taylor Expansion: $\int_{t^{[0]}}^{t^{[1]}} \mathbf{v}(t) d t=\Delta t \mathbf{v}\left(t^{[0]}\right)+\frac{\Delta t^{2}}{2} \mathbf{v}^{\prime}\left(t^{[0]}\right)+\cdots = \Delta t \mathbf{v}\left(t^{[0]}\right)+ \mathcal O(\Delta t^2)$	(Error: $\mathcal O(\Delta t^2)$)
  
- **Implicit Euler** (1st-order accurate): sets the height at $t^{[1]}$ 
  $$
  \int_{t^{[0]}}^{t^{[1]}} \vb{v}(t)\ \mathrm{d}t \approx \Delta t \ \vb{v}(t^{[1]})
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211117220639699.png" alt="image-20211117220639699" style="zoom:50%;" />
  
  Taylor: $\int_{t^{[0]}}^{t^{[1]}} \mathbf{v}(t) d t=\Delta t \mathbf{v}\left(t^{[1]}\right)+\frac{\Delta t^{2}}{2} \mathbf{v}^{\prime}\left(t^{[1]}\right)+\cdots = \Delta t \mathbf{v}\left(t^{[1]}\right)+ \mathcal O(\Delta t^2)$	(Error: $\mathcal O(\Delta t^2)$)
  
- **Mid-point** (2nd-order accurate): sets at $t^{[0.5]}$
  $$
  \int_{t^{[0]}}^{t^{[1]}} \vb{v}(t)\ \mathrm{d}t \approx \Delta t \ \vb{v}(t^{[0.5]})
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211117220959053.png" alt="image-20211117220959053" style="zoom:50%;" />

  Taylor: 
  $$
  \begin{aligned}
  \int_{t^{[0]}}^{t^{[1]}} \mathbf{v}(t) d t&=\int_{t^{[0]}}^{t^{[0.5]}} \mathbf{v}(t)\ \mathrm d t+\int_{t^{[0.5]}}^{t^{[1]}} \mathbf{v}(t)\ \mathrm d t \\
  & = \frac{1}{2} \Delta t \mathbf{v}\left(t^{[0.5]}\right)-\frac{\Delta t^{2}}{2} \mathbf{v}^{\prime\left(t^{[0.5]}\right)}+O\left(\Delta t^{3}\right)+\frac{1}{2} \Delta t \mathbf{v}\left(t^{[0.5]}\right)+\frac{\Delta t^{2}}{2} \mathbf{v}^{\prime\left(t^{[0.5]}\right)}+O\left(\Delta t^{3}\right)\\
  &= \Delta t\ \vb{v}(t^{[0.5]}) + \mathcal O (\Delta t^3)
  \end{aligned}
  $$

- Final Method in this case: **Semi-implicit** (Mid-point)

  Velocity -> Explicit; Position -> Implicit
  $$
  \left\{
  \begin{aligned}
  \vb{v}^{[1]} &= \vb{v}^{[0]} + \Delta t M^{-1}\vb{f}^{[0]}\\
  \vb{x}^{[1]} &= \vb{x}^{[0]} + \Delta t\vb{v}^{[1]}
  \end{aligned}
  \right.
  $$
  Alternative: **Leapfrog Integration**

  v and x not overlap (Mid-points)
  $$
  \left\{
  \begin{aligned}
  \vb{v}^{[0.5]} &= \vb{v}^{[-0.5]} + \Delta t M^{-1}\vb{f}^{[0]}\\
  \vb{x}^{[1]} &= \vb{x}^{[0]} + \Delta t\vb{v}^{[0.5]}
  \end{aligned}
  \right.
  $$
  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118001542996.png" alt="image-20211118001542996" style="zoom:50%;" />

### Type of Forces

- **Gravity Force**: $\vb{f}_{\text{gravity}}^{[0]} = M\vb{g}$
- Drag Force: $\vb{f}^{[0]}_{\text{drag}} = -\sigma \vb{v}^{[0]}$ ($\sigma$ - drag coefficient) -> Reduced by following
- Use a coefficient to replace the drag force: $\vb{v}^{[1]} = \alpha \vb{v}^{[0]}$ ($\alpha $ - **decay coefficient**)

### Translation Only Simulation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118002125525.png" alt="image-20211118002125525" style="zoom: 50%;" />

**Steps**: (Mass $M$ and Timestep $\Delta t$ are user spec var)

- $\vb{f}_i^{[0]} \leftarrow \text{Force} (\vb{x}_i^{[0]}, \vb{v}_i^{[0]})$
- $\vb{f}^{[0]} \leftarrow \sum \vb{f}_i^{[0]}$
- $\vb{v} ^{[1]} \leftarrow \vb{v}^{[0]} + \Delta tM^{-1}\vb{f}^{[0]}$
- $\vb{x}^{[1]} \leftarrow \vb{x}^{[0]} + \Delta t\vb{v}^{[1]}$

## Rotation Motion

### Rotation Representations

#### Rotation Represented by Matrix

$$
\vb{R} =\left[\begin{array}{lll}
r_{00} & r_{01} & r_{02} \\
r_{10} & r_{11} & r_{12} \\
r_{20} & r_{21} & r_{22}
\end{array}\right] 
$$

Suitable in graphics, rotation for vertices

**Not suitable for dynamics**:

- Too much redundancy: 9 elem, 3 dof
- Not intuitive
- Defining its time derivative (rotational vel) is difficult

#### Rotation Represented by Euler Angles

Use 3 axial rotations to represent one general rotation. Each axial rotation uses an angle.

Used in Unity. (the order is rotaion-by-Z / X / Y) Intuitive. 

**Not suitable for dynamics**:

- Lose DOFs in certain statues: Gimbal lock

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118003830863.png" alt="image-20211118003830863" style="zoom:33%;" /> 

- Defining its time derivative is difficult

#### Rotation Represented by Quaternion

Complex multiplications: In the complex system, two numbers represent a 2D point. => Quaternion: i, j, k are imaginary numbers (3D space) Four numbers represent a 3D point (with multiplication and division).

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118004837540.png" alt="image-20211118004837540" style="zoom:63%;" /> 

##### Arithematic

Let $\vb{q} = [s\ \ \vb{v}]$ be a quaternion made of 2 parts: a scalar $s$ and a 3D vector $\vb{v}$ for $\vb{ijk}$

- $a \mathbf{q}=\left[\begin{array}{ll}
  a s & a \mathbf{v}
  \end{array}\right]$ 	Scalar-quaternion Multiplication
- $\mathbf{q}_{1} \pm \mathbf{q}_{2}=\left[s_{1} \pm s_{2} \quad \mathbf{v}_{1} \pm \mathbf{v}_{2}\right]$	Addition/Subtraction (Same as vector)
- $\mathbf{q}_{1} \times \mathbf{q}_{2}=\left[s_{1} s_{2}-\mathbf{v}_{1} \cdot \mathbf{v}_{2} \quad s_{1} \mathbf{v}_{2}+s_{2} \mathbf{v}_{1}+\mathbf{v}_{1} \times \mathbf{v}_{2}\right]$	Multiplication
- $$||\vb{q}|| = \sqrt{s^2 + \vb{v}\cdot\vb{v}}$$	Magnitude

In Unity: provide multiplication, but no addition/subtraction/…; Use w, x, y, z -> s, v

##### Representation

Rotate around $\vb{v}$ by angle $\theta$
$$
\left\{\begin{array}{l}
\mathbf{q}=\left[\begin{array}{ll}
\cos \frac{\theta}{2} & \mathbf{v}
\end{array}\right] \\
\|\mathbf{q}\|=1
\end{array}\right. \Rightarrow
\left\{\begin{array}{l}
\mathbf{q}=\left[\begin{array}{ll}
\cos \frac{\theta}{2} & \mathbf{v}
\end{array}\right] \\
\|\mathbf{v}\|^{2}=\sin ^{2} \frac{\theta}{2}
\end{array}\right.
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118011259709.png" alt="image-20211118011259709" style="zoom:67%;" />

Convertible to the matrix:
$$
\mathbf{R}=\left[\begin{array}{ccc}
s^{2}+x^{2}-y^{2}-z^{2} & 2(x y-s z) & 2(x z+s y) \\
2(x y+s z) & s^{2}-x^{2}+y^{2}-z^{2} & 2(y z-s x) \\
2(x z-s y) & 2(y z+s x) & s^{2}-x^{2}-y^{2}+z^{2}
\end{array}\right]
$$

#### Rotation Representations in Unity

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118011526881.png" alt="image-20211118011526881" style="zoom: 67%;" />

### Rotation Motion

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118011710206.png" alt="image-20211118011710206" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118012010411.png" alt="image-20211118012010411" style="zoom:67%;" />

Use a 3D vector $\mathrm{\boldsymbol {\omega}}$ to denote **angular velocity**:

The dir of $\boldsymbol {\omega}$ -> the axis; The magnitude of $\boldsymbol {\omega}$ -> the speed (sim to the representation of quaternion)

#### Torque and Inertia

##### Torque

(Original state: $\vb{r}_i$ -> Rotated: $\vb{Rr}_i$, $\vb{f}_i$ is a force)

The rotational equiv of a force, describing the rotational **tendency** caused by a force.

$\boldsymbol \tau_i$: perpendicular to both $\vb{Rr}_i$ and $\vb{f}_i$; proportional to $||\vb{Rr}_i||$ and $||\vb{f}_i||$; proportional to $\sin \theta$ (the angle of the two vectors)
$$
\boldsymbol{\tau}_i \leftarrow \vb{Rr}_i \times \vb{f}_i
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123174242025.png" alt="image-20211123174242025" style="zoom:67%;" />

##### Inertia

Describes the **resistance** to rotational tendency caused by torque (not const)

Left side (heavier point far away from the torque) has higher resistance (inertia) to the rotational tendency, slower rotation 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123174709048.png" alt="image-20211123174709048" style="zoom: 50%;" />

Ref state inertia, change with rotation (dep on pose). But no need to re-compute every time
$$
\mathbf{I}_{\mathbf{r e f}}=\sum m_{i}\left(\mathbf{r}_{i}^{\mathrm{T}} \mathbf{r}_{i} \mathbf{1}-\mathbf{r}_{i} \vb{r}_{i}^{\mathrm{T}}\right)
$$
($\vb{1}$ - 3x3 identity matrix)
$$
\begin{aligned} \mathbf{I} &=\sum m_{i}\left(\mathbf{r}_{i}^{\mathrm{T}} \mathbf{R}^{\mathrm{T}} \mathbf{R} \mathbf{r}_{i} \mathbf{1}-\mathbf{R} \mathbf{r}_{i} \boldsymbol{r}_{i}^{\mathrm{T}} \mathbf{R}^{\mathrm{T}}\right) \\ &=\sum m_{i}\left(\mathbf{R} \mathbf{r}_{i}^{\mathrm{T}} \mathbf{r}_{i} \mathbf{1} \mathbf{R}^{\mathrm{T}}-\mathbf{R} \mathbf{r}_{i} \boldsymbol{r}_{i}^{\mathrm{T}} \mathbf{R}^{\mathrm{T}}\right) \\ &=\sum m_{i} \mathbf{R}\left(\mathbf{r}_{i}^{\mathrm{T}} \mathbf{r}_{i} \mathbf{1}-\mathbf{r}_{i} \boldsymbol{r}_{i}^{\mathrm{T}}\right) \mathbf{R}^{\mathrm{T}} \\ &=\mathbf{R} \mathbf{I}_{\mathbf{r e f}} \mathbf{R}^{\mathrm{T}} \end{aligned}
$$

##### Use torque to represent

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118103522555.png" alt="image-20211118103522555" style="zoom:50%;" />

- **Torque** on a spec point: $\boldsymbol{\tau}_i = (\vb{Rr}_i)\times \vb{f}_i$

  Total torque: $\boldsymbol{\tau}_{i} = \sum\boldsymbol{\tau}_{i}$ 

- The rotational equivalent of **mass** is called **inertia** $\vb{I}$ (Result: 3x3):

  Reference inertia: $\vb{I}_{\mathrm{ref}} = \sum m_i(\vb{r}_i^{\mathrm{T}}\vb{r}_i\vb{1} - \vb{r}_i\vb{r}_i^{\mathrm{T}}) $

  Current inertia: $\vb{I} = \vb{R} \vb{I}_{\mathrm{ref}} \vb{R}^{\mathrm{T}}$

## Rigid Body Simulation

### Translational and Rotational Motion

- Translation (Linear)

  States: velocity $\vb{v}$ and position $\vb{x}$ (`transform.position` in Unity)

  Physical Quantities: mass $M$ and force $\vb{f}$
  $$
  \left\{\begin{array}{l}
  \mathbf{v}^{[1]}=\mathbf{v}^{[0]}+\Delta t M^{-1} \mathbf{f}^{[0]} \\
  \mathbf{x}^{[1]}=\mathbf{x}^{[0]}+\Delta t \mathbf{v}^{[1]}
  \end{array}\right.
  $$

- Rotational (Angular): Better normalize when $||\vb{q}||\ne1$ (in Unity automatically)

  States: angular velocity $\boldsymbol{\omega}$ and quaternion $\vb{q}$ (`transform.rotation` in Unity)

  Physical Quantites: inertia $\vb{I}$ and torque $\boldsymbol{\tau}$ 
  $$
  \left\{\begin{array}{l}
  \boldsymbol{\omega}^{[1]}=\boldsymbol{\omega}^{[0]}+\Delta t\left(\mathbf{I}^{[0]}\right)^{-1} \mathbf{\tau}^{[0]} \\
  \mathbf{q}^{[1]}=\mathbf{q}^{[0]}+\left[0 \quad \frac{\Delta t}{2} \boldsymbol{\omega}^{[1]}\right] \times \mathbf{q}^{[0]}
  \end{array}\right.
  $$

### Rigid Body Simulation Process

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118104825451.png" alt="image-20211118104825451" style="zoom:67%;" />

In Unity: No 3x3 matrices, only 4x4 (use 4x4 and set the last col / line ); Provide `.inverse` to inverse; …

### Implementation

In practice, we update the same state var $\vb{s = \{ v,x},\boldsymbol{\omega},\vb{q\}}$  

**Issues**

- Translation is easier, code translation first
- Using a const $\boldsymbol{\omega}$ first while testing update $\vb{q}$, in this case the object will spin constantly
- Gravity does NOT cause torque (except for air drag force)



# Lecture 4 Rigid Body Contacts 

## Particle Collision Detection and Response

### Distance Functions

#### Signed Distance Function

Use a signed distance func $\phi(\vb{x})$ to define the distance indicating which side as well (corresponding to 0 surface)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123201654431.png" style="zoom:50%;" />

**Examples**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123201917848.png" alt="image-20211123201917848" style="zoom: 50%;" />
$$
\phi(\vb{x}) = (\vb{x-p})\cdot \vb{n}\quad\qquad 
\phi(\vb{x}) = ||\vb{x-c}|| - r\quad\qquad
\phi(\vb{x}) = \sqrt{||\vb{x-p}||^2 - ((\vb{x-p})\cdot \vb{n})^2} - \vb{r}
$$

#### Intersection of Signed Distance Functions

=> Bool operations

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123205109282.png" alt="image-20211123205109282" style="zoom: 50%;" />

if $\phi_0(\vb{x})< 0$ and $\phi_1(\vb{x})< 0$ and $\phi_2(\vb{x})< 0$
	then inside
	$\phi(\vb{x})=\max(\phi_0{\vb{x}}, \phi_1(\vb{x}), \phi_2(\vb{x}))$	// all val are negative, the max one is the closest
else
	$\phi(\vb{x}) = ?$ 	// not relavent (no collision)

#### Union of Signed Distance Functions

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123211143939.png" alt="image-20211123211143939" style="zoom:50%;" />



if $\phi_0(\vb{x})< 0$ or $\phi_1(\vb{x})< 0$ 
​	then inside
​	$\phi(\vb{x}) \approx \min(\phi_0(\vb{x}),\phi_1(\vb{x}))$	// approximate -> correct near outer boundary
else outside
​	$\phi(\vb{x}) = \min(\phi_0(\vb{x}), \phi_1(\vb{x}))$ 

-> We can consider collision detection with the union of two objects as collision detection with two separate objects

### Penalty methods

(Implicit integration is better)

#### Quadratic Penalty Method

Check if collide - Yes -> Apply a force at the point (in the next update)

For **quadratic** penalty (strength) potential, the force is **linear**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123212648589.png" alt="image-20211123212648589" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123212726821.png" alt="image-20211123212726821" style="zoom:50%;" />

Problem: Already inside -> cause artifacts 

=> Add **buffer** help less the penetration issue (cannot strictly prevent)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123212952453.png" alt="image-20211123212952453" style="zoom:50%;" />

Problem: $k$ too low -> buffer not work well; $k$ too high -> too much force generated (overshooting)

#### Log-Barrier Penalty Method

Ensures that the force can be large enough, but assumes $\phi(\vb{x})< 0$ never happens => By adjusting $\Delta t$

Always apply the penalty force: $\vb{f} \leftarrow \rho \frac{1}{\phi(\vb{x})} \vb{N}$ ($\rho$ - Barrier strength)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123213631306.png" alt="image-20211123213631306" style="zoom:50%;" />

Problems: cannot prevent overshooting when very close to the surface; when penatration occurs, the penatration will be higher and higher (-> smaller step size -> higher costs)

=> Log-Barrier limited within a buffer as well to solve

Frictional contacts are difficult to handle

### Impulse method

Update the vel and pos as the collision occurs

Changing the **position**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123223036097.png" alt="image-20211123223036097" style="zoom:50%;" />

Changing the **velocity**: ($\mu_{\vb{N}}$ - bounce coefficient, $\in [0,1]$, $a$ - frictional decay of vel)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123224436596.png" alt="image-20211123224436596" style="zoom:60%;" /><img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123224520049.png" alt="image-20211123224520049" style="zoom:40%;" />

$a$ should be minimized but not violating Coulomb’s law
$$
\left\|\mathbf{v}_{\mathbf{T}}^{\text {new }}-\mathbf{v}_{\mathbf{T}}\right\| \leq \mu_{\mathbf{T}}\left\|\mathbf{v}_{\mathbf{N}}^{\text {new }}-\mathbf{v}_{\mathbf{N}}\right\|\\
(1-a)\left\|\mathbf{v}_{\mathbf{T}}\right\| \leq \mu_{\mathbf{T}}\left(1+\mu_{\mathbf{N}}\right)\left\|\mathbf{v}_{\mathbf{N}}\right\|
$$
Therefore:
$$
a \longleftarrow \max (1-\underbrace{\mu_{\vb{T}}\left(1+\mu_{\vb{N}}\right)\left\|\mathbf{v}_{\vb{N}}\right\| /\left\|\mathbf{v}_{\vb{T}}\right\|}_{\text{dynamic friction}},\underbrace{0}_{\text{static friction}})
$$
Can precisely control the friction effects

## Rigid Collision Detection and Response by Impulse

### Rigid Body Collision Detection

When the body is made of many vertices, test each vertex: $\vb{x}_i \leftarrow \vb{x}+ \vb{Rr}_i$ (from the mass center to the vertices)

=> detection: transverse every point if $\phi(\vb{x})<0$

### Rigid Body Collision Response by Impulse

Vertex $i$: ($\vb{v}$ - linear vel; $\boldsymbol{\omega}$ - angular vel)
$$
\left\{
\begin{aligned}
\vb{x}_i &\leftarrow \vb{x}+ \vb{Rr}_i&\quad(\text{Position})\\
\vb{v}_i &\leftarrow \vb{v}+\boldsymbol{\omega}\times \vb{Rr}_i&\quad (\text{Velocity})
\end{aligned}
\right.
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211123230332712.png" alt="image-20211123230332712" style="zoom:50%;" />

But cannot modify $\vb{x}_i$ and $\vb{v}_i$ directly since they are not state var. (in Lec.3: 4 var are vel of mass center / pos of mass center / rotational pose / angular vel) 

**Applying an impulse** $\vb{j}$ at vertex $i$: ($\Delta \vb{v} = \frac{\vb{j}}{M}$ (Newton’s Law); $\vb{Rr}_i \times \vb{j}$ - torque induced by $\vb{j}$)
$$
\left\{
\begin{aligned}
\vb{v}^{\text{new}} &\leftarrow \vb{v}+ \frac{1}{M}\vb{j}\\
\boldsymbol{\omega} ^{\text{new}} &\leftarrow \boldsymbol{\omega} + \vb{I}^{-1} (\vb{Rr}_i \times \vb{j})
\end{aligned}
\right.
$$

$$
\begin{aligned}
\Rightarrow \vb{v}^{\text{new}}_{i} &= \vb{v}^{\text{new}} + \boldsymbol{\omega}^{\text{new}}\times\vb{Rr}_i\\
&=\mathbf{v}+\frac{1}{M} \mathbf{j}+\left(\boldsymbol{\omega}+\mathbf{I}^{-1}\left(\mathbf{R} \mathbf{r}_{i} \times \mathbf{j}\right)\right) \times \mathbf{R} \mathbf{r}_{i}\\
&=\mathbf{v}_{i}+\frac{1}{M} \mathbf{j}-\left(\mathbf{R} \mathbf{r}_{i}\right) \times\left(\mathbf{I}^{-1}\left(\mathbf{R} \mathbf{r}_{i} \times \mathbf{j}\right)\right)
\end{aligned}
$$

**Cross Product as a Matrix Product ** 

Convert the cross prod $\vb{r}\times$ into a matrix prod $\vb{r}^*$ 
$$
\mathbf{r} \times \mathbf{q}=\left[\begin{array}{l}
r_{y} q_{z}-r_{z} q_{y} \\
r_{z} q_{x}-r_{x} q_{z} \\
r_{x} q_{y}-r_{y} q_{x}
\end{array}\right]=\left[\begin{array}{ccc}
0 & -r_{z} & r_{y} \\
r_{z} & 0 & -r_{x} \\
-r_{y} & r_{x} & 0
\end{array}\right]\left[\begin{array}{l}
q_{x} \\
q_{y} \\
q_{z}
\end{array}\right]=\mathbf{r}^{*} \mathbf{q}
$$
In our case:
$$
\Rightarrow \vb{v}_i^{\text{new}} = \mathbf{v}_{i}^{\text {new }}=\mathbf{v}_{i}+\frac{1}{M} \mathbf{j}-\left({\mathbf{R}} \mathbf{r}_{i}\right)^{*} \mathbf{I}^{-1}\left(\mathbf{R} \mathbf{r}_{i}\right)^{*} \mathbf{j}
$$
Therefore, replace with some matrix $\vb{K}$. Finally $\vb{j}$ can be computed with the following equations.
$$
\Rightarrow 
\mathbf{v}_{i}^{\text {new }}-\mathbf{v}_{i}=\mathbf{K j} \\
\mathbf{K} \leftarrow \frac{1}{M} \mathbf{1}-\left(\mathbf{R} \mathbf{r}_{i}\right)^{*} \mathbf{I}^{-1}\left(\mathbf{R} \mathbf{r}_{i}\right)^{*}
$$

### Implementation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124001811694.png" alt="image-20211124001811694" style="zoom:80%;" />

**Other details**:

- For several vertices in collision, use their **average**
- Can **decrease the restitution** $\mu_{\vb{N}}$ to reduce oscillation
- Don’t update the position: not linear

## Shape Matching

### Basic Idea

Allow each vertex to have its own velocity, so it can move by itself

- Move vertices independently by its velocity, with collision and friction being handled (use the impulse method for every point)
- Enforce the rigidity constraint to become a rigid body again (IMPORTANT)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124002525946.png" alt="image-20211124002525946" style="zoom:50%;" />

### Mathematical Formulation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124002656804.png" alt="image-20211124002656804" style="zoom:50%;" />

Want to find $\vb{c}$ and $\vb{R}$ ($\vb{c}$ - mass center): want the final rigid body (a square) close enough to the trapozoid

($\vb{A}$ - a matrix, not only corresponding to rotation, $\sum \vb{Ar}_i = 0$ since the mass center was set to 0; $E$ - The objective, $= \frac{1}{2}\left\|\mathbf{c}+\mathbf{A} \mathbf{r}_{i}-\mathbf{y}_{i}\right\|^{2}$)
$$
\begin{aligned}
\{\mathbf{c}, \mathbf{R}\}&=\operatorname{argmin} \sum_{i} \frac{1}{2}\left\|\mathbf{c}+\mathbf{R} \mathbf{r}_{i}-\mathbf{y}_{i}\right\|^{2} \\
\Rightarrow 
\{\mathbf{c}, \mathbf{A}\}&=\operatorname{argmin} \sum_{i} \frac{1}{2}\left\|\mathbf{c}+\mathbf{A} \mathbf{r}_{i}-\mathbf{y}_{i}\right\|^{2}

\end{aligned}
$$
For mass center $\vb{c}$ and matrix $\vb{A}$ (Find derivatives):
$$
\frac{\partial E}{\partial \mathbf{c}}=\sum_{i} \mathbf{c}+\cancel{\mathbf{A} \mathbf{r}_{i}}-\mathbf{y}_{i}=\sum_{i} \mathbf{c}-\mathbf{y}_{i}=\mathbf{0}\\
\Rightarrow \vb{c} = \frac{1}{N}\sum_i\vb{y}_i \quad (\text{average})
$$

$$
\frac{\partial E}{\partial \mathbf{A}}=\sum_{i}\left(\mathbf{c}+\mathbf{A} \mathbf{r}_{i}-\mathbf{y}_{i}\right) \mathbf{r}_{i}^{\mathrm{T}}=\mathbf{0}\\
\mathbf{A}=\left(\sum_{i}\left(\mathbf{y}_{i}-\mathbf{c}\right) \mathbf{r}_{i}^{\mathrm{T}}\right)\left(\sum_{i} \mathbf{r}_{i} \mathbf{r}_{i}^{\mathrm{T}}\right)^{-1}  \xlongequal{\text{Polar Decomposition}}\vb{\underbrace{R}_{\text{rotation}}\, \underbrace{S}_{\text{deformation}}}
$$

**Remind**: Singular value decomposition: $\vb{A} = \vb{UDV}^{\mathrm{T}}$ (rotaion, scaling and rotation)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109165409060.png" alt="image-20211109165409060" style="zoom:70%;" />

Rotate the object back before the final rotation: $\vb{A} = (\vb{UV}^{\mathrm{T}})(\vb{VDV}^{\mathrm{T}}) = \vb{RS}$ (Local deformation: $\vb{VDV}^{\mathrm{T}} = \vb{S}$)

![image-20211124004551525](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124004551525.png)

### Implementation

Physical quantities are attached to each vertex, not to the entire body.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124005104997.png" alt="image-20211124005104997" style="zoom:80%;" />

(The function of $\mathrm{Polar}(\vb A)$ is provided, ultilizing the polar deposition technique)

**Properties**:

- Easy to **implement** and **compatible with other nodal systems**: cloth, soft bodies, particle fluids, …
- Difficult to **strictly enforce friction and other goals**. The rigidification process will destroy them (may require iter)
- More suitable when the **friction accuracy is unimportant**, i.e., buttons on clothes



# Lecture 5 Physics-Based Cloth Simulation

## A Mass-Spring System

### Spring Systems

#### An Ideal Spring

Satisfies Hooke’s Law: The spring force tries to restore the rest length ($k$ - Spring Stiffness)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130163240282.png" alt="image-20211130163240282" style="zoom: 50%;" />
$$
E(x) = \frac{1}{2}k(x-L)^2\ ;\quad f(x) = -\frac{\mathrm{d}E}{\mathrm{d}x} = -k(x-L)
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130163615639.png" alt="image-20211130163615639" style="zoom: 50%;" />
$$
E(\vb{x}) = \frac{1}{2} k(\|\vb{x}_i - \vb{x}_j\| -L)\\
\vb{f}_i = -\vb{f}_j\quad
\left\{\begin{aligned}
\mathbf{f}_{i}(\mathbf{x})=-\grad_{i} E=-k\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|} \\
\mathbf{f}_{j}(\mathbf{x})=-\grad_{j} E=-k\left(\left\|\mathbf{x}_{j}-\mathbf{x}_{i}\right\|-L\right) \frac{\mathbf{x}_{j}-\mathbf{x}_{i}}{\left\|\mathbf{x}_{j}-\mathbf{i}_{j}\right\|} 
\end{aligned}\right.
$$

#### Multiple Springs

The energies and forces can be simply summed up

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130164434830.png" alt="image-20211130164434830" style="zoom:50%;" />
$$
E=\sum_{j=0}^{3} E_{j}=\sum_{j=0}^{3}\left(\frac{1}{2} k\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-L\right)^{2}\right) \\
\mathbf{f}_{i}=-\nabla_{i} E=\sum_{j=0}^{3}\left(-k\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|}\right)
$$

### Structures in Simulations

#### Structured Spring Networks

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130164711386.png" alt="image-20211130164711386" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130164742643.png" style="zoom:50%;" />

#### Unstructured Spring Networks

##### **Unstructured triangle mesh**

-> the edges into spring networks (usually in cloth simulations)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130165041811.png" style="zoom:50%;" /> Blue lines for bending resistance (every neighboring triangle pair)

##### Triangle Mesh Representation

Two arrays: **Vertex** & **Triangle lines**

- Vertex list: {$\vb{x}_0$, $\vb{x}_1$, $\vb{x}_2$, $\vb{x}_3$, $\vb{x}_4$} (3D vectors)
- Triangle list: `{1, 2, 3, 0, 1, 3, 0, 3, 4}` (Index triples) => (`{1, 2, 3}` for Triangle 0; `{0, 1, 3}` for Triangle 1; …)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130165206242.png" alt="image-20211130165206242" style="zoom:50%;" />

Not only edges but also an inner one (each triangle has 3 edges but there are repeated ones)

##### Topological Construction

Sort triangle edge triples: edge vertex index 0 & 1 and triangle index (index 0 < index)

- Triple list: 

  ``` c
  {{1, 2, 0}, {2, 3, 0}, {1, 3, 0},	// Green triangle: {1, 2} for edge index; {0} at the end for triangle index
   {0, 1, 1}, {1, 3, 1}, {0, 3, 1},
   {0, 3, 2}, {3, 4, 2}, {0, 4, 2}}
  ```

- Sorted triple list: `{{0, 1, 1}, {0, 3, 1}, {0, 3, 2}, {0, 4, 2}, {1, 2, 0}, {1, 3, 0}, {1, 3, 1}, {2, 3, 0},  {3, 4, 2}}`

  Repeated edges `{1, 3, 0}|{1, 3, 1}` & `{0, 3, 1}|{0, 3, 2}` shows in the neighbor => eliminate

- **Final edge list:** `{{0, 1}, {0, 3}, {0, 4}, {1, 2}, {1, 3}, {2, 3}, {3, 4}}`

  **Neighboring triangle list**: `{{1, 2}, {0, 1}}` (for bending) (or use neighboring edge list)

### Integrators

#### Explicit Integration

##### Scheme

(Notations: $m_i$ - Mass of vertex $i$; $E$ - Edge list; $L$ - Edge length list (pre-computed))

- For every vertex: 
  - $\vb{f}_i \leftarrow \text{Force}(\vb{x}_i, \vb{v}_i)$
  - $\vb{v}_i \leftarrow \vb{v}_i + \Delta tm_i^{-1}\vb{f}_i$
  - $\vb{x}_i \leftarrow \vb{x}_i + \Delta t\vb{v}_i$
- To compute Spring Forces: (For every spring $e$)
  - $i\leftarrow E[e][0]$  (Spring index $[e]$ and vertex index $[0]$)
  - $j \leftarrow E[e][1]$
  - $L_e \leftarrow L[e]$
  - $\vb{f}\leftarrow -k(\|\vb{x}_i - \vb{x}_j\| - L_e) \frac{\vb{x}_i - \vb{x}_j}{\|\vb{x}_i - \vb{x}_j\|}$ 
  - $\vb{f}_i\leftarrow \vb{f}_i + \vb{f}$
  - $\vb{f}_j\leftarrow \vb{f}_j + \vb{f}$ 

##### Problem

Overshooting: when $k $ and/or $\Delta t$ is too large

Naive solution: Reduce $\Delta t$ => Slow down the simulation

#### Implicit Integration

##### General Scheme (Euler Method)

Integrate both $\vb{x}$ and $\vb{v}$ implicitly ($\vb{M}\in \R^{3N \times 3N}$ - Mass matrix (usually diagonal))
$$
\left\{\begin{array}{l}
\mathbf{v}^{[1]}=\mathbf{v}^{[0]}+\Delta t \mathbf{M}^{-1} \mathbf{f}^{[1]} \\
\mathbf{x}^{[1]}=\mathbf{x}^{[0]}+\Delta t \mathbf{v}^{[1]}
\end{array}\right.\quad
\text{or}\quad
\left\{\begin{array}{l}
\mathbf{x}^{[1]}=\mathbf{x}^{[0]}+\Delta t \mathbf{v}^{[0]}+\Delta t^{2} \mathbf{M}^{-1} \mathbf{f}^{[1]} \\
\mathbf{v}^{[1]}=\left(\mathbf{x}^{[1]}-\mathbf{x}^{[0]}\right) / \Delta t
\end{array}\right.
$$
$\vb{v}^{[1]}$ & $\vb{x}^{[1]}$ are unknown for the current time step => Find the x and v:

Assume $\vb{f}$ dep only on $\vb{x}$ (homonomic): Solve the eqn (Problem: $\vb{f}$ may not be a linear function)
$$
\mathbf{x}^{[1]}=\mathbf{x}^{[0]}+\Delta t \mathbf{v}^{[0]}+\Delta t^{2} \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}^{[1]}\right)
$$
The equation equiv to the following (where $\|\vb{x}\|^2_\vb{M} = \vb{x}^{\mathrm{T}}\vb{Mx}$) => **Optimization prob** => Numerical schemes (Usually only conservative forces can use this energy)
$$
\mathbf{x}^{[1]}=\operatorname{argmin} F(\mathbf{x}) \quad \text { for } \quad F(\mathbf{x})=\frac{1}{2 \Delta t^{2}}\left\|\mathbf{x}-\mathbf{x}^{[0]}-\Delta t \mathbf{v}^{[0]}\right\|_{\mathbf{M}}^{2}+E(\mathbf{x})
$$
Because: (applicable for every system; The first order der of $F(\vb{x})$ reaches 0 for the min pt.)
$$
\nabla F\left(\mathbf{x}^{[1]}\right)=\frac{1}{\Delta t^{2}} \mathbf{M}\left(\mathbf{x}^{[1]}-\mathbf{x}^{[0]}-\Delta t \mathbf{v}^{[0]}\right)-\mathbf{f}\left(\mathbf{x}^{[1]}\right)=\mathbf{0} \Rightarrow \mathbf{x}^{[1]}-\mathbf{x}^{[0]}-\Delta t \mathbf{v}^{[0]}-\Delta t^{2} \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}^{[1]}\right)=\mathbf{0}
$$

##### The Optimization Problem

###### Newton-Raphson Method

Solving optimization problem: ${x}^{[1]} = \operatorname{argmin} F(x)$ ($F(x)$ is Lipschitz continuous)

Given a current $x^{(k)}$ we approx the goal by $0 = F'(x) \approx F'(x^{(k)}) + F''(x^{(k)})(x-x^{(k)})$ (Taylor Expansion)

(For 2D: $\vb{0} = \grad F(\vb{x}) \approx \grad F(\vb{x}^{(k)}) + \frac{\partial F^2 (\vb{x}^{(k)})}{\partial \vb{x}^2} (\vb{x}-\vb{x}^{(k)})$) 

**Steps**:

- Initialize $x^{(0)}$

- For $k = 0\ ...\ k$

  ​	$\Delta x \leftarrow -(F''(x^{(k)}))^{-1} F'(x^{(k)})$	(For 2D: $\Delta \mathbf{x} \leftarrow-\left(\frac{\partial F^{2}\left(\mathbf{x}^{(k)}\right)}{\partial \mathbf{x}^{2}}\right)^{-1} \grad F\left(\mathbf{x}^{(k)}\right)$)

  ​	$x^{(k+1)} \leftarrow x^{(k)} + \Delta x$

  ​	If $|\Delta x|$ is small 	then break

- $\vb{x}^{[1]} \leftarrow \vb{x}^{(k+1)}$ 

Newton’s Method finds extremum, but it can be min or max => finds 2nd order derivative (at local min: $F''(x^*) > 0$; at max: $F''(x^*)<0$)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130210036973.png" alt="image-20211130210036973" style="zoom:50%;" />

For a function which second order derivative always larger than 0 ($F''(x)>0$ everywhere) => $F(x)$ has only one minimum

###### Simulation by Newton’s Method

For simulation: $F(\mathbf{x})=\frac{1}{2 \Delta t^{2}}\left\|\mathbf{x}-\mathbf{x}^{[0]}-\Delta t \mathbf{v}^{[0]}\right\|_{M}^{2}+E(\mathbf{x})$

- Derivative: 
  $$
  \nabla F\left(\mathbf{x}^{(k)}\right)=\frac{1}{\Delta t^{2}} \mathbf{M}\left(\mathbf{x}^{(k)}-\mathbf{x}^{[0]}-\Delta t \mathbf{v}^{[0]}\right)-\mathbf{f}\left(\mathbf{x}^{(k)}\right)
  $$

- Force Hessian Matrix: ($\vb{H}(\vb{x}^{(k)})$ - Energy Hessian)  
  $$
  \frac{\partial^{2} F\left(\mathbf{x}^{(k)}\right)}{\partial \mathbf{x}^{2}}=\frac{1}{\Delta t^{2}} \mathbf{M}+\mathbf{H}\left(\mathbf{x}^{(k)}\right)
  $$

**Steps**:

- Initialize $\vb{x}^{(0)}$, often as $\vb{x}^{[0]}$ or $\vb{x}^{[0] }+\Delta t\vb{v}^{[0]}$

- For $k = 0 \ ...\ k$:

  ​	Solve $\left(\frac{1}{\Delta t^{2}} \mathbf{M}+\mathbf{H}\left(\mathbf{x}^{(k)}\right)\right) \mathbf{x}=\left(-\frac{1}{\Delta t^{2}} \mathbf{M}\left(\mathbf{x}^{(k)}-\mathbf{x}^{[0]}- \Delta t \mathbf{v}^{[0]}\right)+\mathbf{f}\left(\mathbf{x}^{(k)}\right)\right.$ 

  ​	$\vb{x}^{(k+1)} \leftarrow \vb{x}^{(k)} + \Delta \vb{x}$

  ​	If $\|\Delta \vb{x}\|$ is small	then break

- $\vb{x}^{[1]}\leftarrow \vb{x}^{(k+1)}$

- $\vb{v}^{[1]}\leftarrow (\vb{x}^{[1]} - \vb{x}^{[0]})/ \Delta t$ 

###### Spring Hessian

Hessian matrix is a second order derivative (sim to $\partial^2 F / \partial \vb{x}^2$) => if p.d. so that the **ONLY min point** is found and has no maximum (sufficient but not neccesary condition)
$$
\mathbf{H}(\mathbf{x})=\sum_{e=\{i, j\}}\left[\begin{array}{cc}
\frac{\partial^{2} E}{\partial \mathbf{x}_{i}^{2}} & \frac{\partial^{2} E}{\partial \mathbf{x}_{i} \partial \mathbf{x}_{j}} \\
\frac{\partial^{2} E}{\partial \mathbf{x}_{i} \partial \mathbf{x}_{j}} & \frac{\partial^{2} E}{\partial \mathbf{x}_{j}^{2}}
\end{array}\right]=\sum_{e=\{i, j\}}\left[\begin{array}{cc}
\mathbf{H}_{e} & -\mathbf{H}_{e} \\
-\mathbf{H}_{e} & \mathbf{H}_{e}
\end{array}\right]
$$
The matrix in the last part is 3N x 3N, every vertex is a 3D vector. The first $\vb{H}_e$ is at $(i,i)$, the $-\vb{H}_e$ in the first line is at $(i,j)$, …

Positive Definite: Dep on every $\vb{H}_e$ (3 x 3): @ Lec.2, P48
$$
\mathbf{H}_{e}=k \frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}+k\left(1-\frac{L}{\left\|\mathbf{x}_{i j}\right\|}\right)\left(\mathbf{I}-\frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}\right) \ ;\quad \vb{x}_{ij} = \vb{x}_i - \vb{x}_j
$$
where $k \frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}$ & $\left(\mathbf{I}-\frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}\right)$ are already s.p.d. (Proof by multiplying by $\vb{v}^{\mathrm{T}}$ and $\vb{v}$ at both sides)

But $\left(1-\frac{L}{\left\|\mathbf{x}_{i j}\right\|}\right)$ can be negative when $\|\vb{x}_{ij}\| < L_e$ (when compressed) 

**Conclusion**: when stretched $\vb{H}_e$ is s.p.d. and when compressed $\vb{H}_e$ may not be s.p.d. => $\vb {A}$ may not be s.p.d. either
$$
\mathbf{A}=\frac{1}{\Delta t^{2}} \mathbf{M}+\mathbf{H}(\mathbf{x})=\underbrace{\frac{1}{\Delta t^{2}} \mathbf{M}}_{\text{s.p.d.}}+\sum_{e=\{i, j\}}\underbrace{\left[\begin{array}{ccc}
\ddots & \vdots & \vdots &  \\
& \mathbf{H}_{e} & -\mathbf{H}_{e} & \\
& -\mathbf{H}_{e} & \mathbf{H}_{e} & \\
 & & & \ddots
\end{array}\right]}_{\text{may not be s.p.d.}}
$$
(for smaller $\Delta t$ => more p.d., actually sim to explicit integration with smaller time step => more stable)

**Positive Definiteness of Hessian**

When a spring is compressed, the spring Hessian may not be positive definite => Multiple minima

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211130233603769.png" alt="image-20211130233603769" style="zoom:50%;" />

Only in 2D/3D: In 1D: $E(x) = \frac{1}{2}k(x-L)^2$ and $E''(x) = k>0$ 

**Enforcement of P.D.**

Some linear solver may require $\vb{A}$ must be p.d. in $\vb{A}\Delta \vb{x} = \vb{b}$ 

Solution:

- Drop the ending term when $\|\vb{x}_{ij} \|<L_e$: 
  $$
  \mathbf{H}_{e}=k \frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}+
  k\cancel{\left(1-\frac{L}{\left\|\mathbf{x}_{i j}\right\|}\right)\left(\mathbf{I}-\frac{\mathbf{x}_{i j} \mathbf{x}_{i j}^{\mathrm{T}}}{\left\|\mathbf{x}_{i j}\right\|^{2}}\right)}
  $$

- *Choi and Ko. 2002. Stable But Responive Cloth. TOG (SIGGRAPH)*

##### Linear Solvers

Solving $\vb{A}\Delta \vb{x} = \vb{b}$

###### Jocobi Method

- $\Delta \vb{x} \leftarrow 0$

  For $k = 0 \ ... \ k$:

  ​	$\vb{r} \leftarrow \vb{b} - \vb{A}\Delta \vb{x}$					// Residual error

  ​	if $\|r\|< \varepsilon$ 	then break	  // Convergence condition $\varepsilon$

  ​	$\Delta \vb{x }\leftarrow \Delta \vb{x }+ \alpha \vb{D}^{-1}\vb{r}$ 		 // Update by $\vb{D}$, the diagonal of $\vb{A}$ 

vanilla Jocobi method ($\alpha = 1$) has a tight convergence req on $\vb{A}$: Diagonal Dominant

###### Other Solvers

- Direct solvers (LU / LDLT / Cholesky / …) – [Intel MKL PARDISO]
  - One shot, expensive but worthy if need exact sol
  - Little restriction on $\vb{A}$
  - Mostly suitable on CPUs
- Iterative solvers
  - Expensive to solve exactly, but controllable
  - Convergence restriction on $\vb{A}$, typically positive definiteness
  - Suitable on both CPUs and GPUs
  - Easy to implement
  - Accelerable: Chebyshev, Nesterov, <u>Conjugate Gradient</u> …

##### After-Class Reading

- *Baraff and Witkin. 1998. Large Step in Cloth Simulation. SIGGRAPH.*

  > One of the first papers using implicit integration. The paper proposes to use only one Newton iteration, i.e., solving only one linear system. This practice is fast, but can fail to converge.

## Bending and Locking Issues

### The Bending Spring Issue

A bending spring offers little resistance when cloth is nearly planar, since its length barely changes.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207150345778.png" alt="image-20211207150345778" style="zoom:50%;" />

### A Dihedral Angle Model

Every vertex of the 4 will be under a force as a function of $\theta$: $\vb{f}_i  = f(\theta) \vb{u}_i$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207150527413.png" alt="image-20211207150527413" style="zoom:50%;" />

- $\vb{u}_1$ and $\vb{u}_2$ should be in the normal directions $\vb{n}_1$ & $\vb{n}_2$ 
- Bending doesn’t stretch the edge $\vb{u}_4 - \vb{u}_3$  should be orthogonal to the edge (in the span of $\vb{n}_1$ & $\vb{n}_2$) 
- Newton: $\vb{u}_1 + \vb{u}_2 + \vb{u}_3 + \vb{u}_4 = 0$, means $\vb{u}_3 $ & $\vb{u}_4$ are in the span of $\vb{n}_1$ & $\vb{n}_2$ 

**Conclusion**:

- $\vb{u}_1 = \|\vb{E}\|\frac{\vb{N}_1}{\|\vb{N}_1\|^2}$ 
- $\vb{u}_2 = \|\vb{E}\|\frac{\vb{N}_2}{\|\vb{N}_2\|^2}$
- $\mathbf{u}_{3}=\frac{\left(\mathbf{x}_{1}-\mathbf{x}_{4}\right) \cdot \mathbf{E}}{\|\mathbf{E}\|} \frac{\mathbf{N}_{1}}{\left\|\mathbf{N}_{1}\right\|^{2}}+\frac{\left(\mathbf{x}_{2}-\mathbf{x}_{4}\right) \cdot \mathbf{E}}{\|\mathbf{E}\|} \frac{\mathbf{N}_{2}}{\left\|\mathbf{N}_{2}\right\|^{2}}$ 
- $\mathbf{u}_{4}=-\frac{\left(\mathbf{x}_{1}-\mathbf{x}_{3}\right) \cdot \mathbf{E}}{\|\mathbf{E}\|} \frac{\mathbf{N}_{1}}{\left\|\mathbf{N}_{1}\right\|^{2}}-\frac{\left(\mathbf{x}_{2}-\mathbf{x}_{3}\right) \cdot \mathbf{E}}{\|\mathbf{E}\|} \frac{\mathbf{N}_{2}}{\left\|\mathbf{N}_{2}\right\|^{2}}$ 

$\vb{N}_1 = (\vb{x}_1 - \vb{x}_3)\times(\vb{x}_1 - \vb{x}_4)$ ;  $\vb{N}_2 = (\vb{x}_2 - \vb{x}_4)\times(\vb{x}_2-\vb{x}_3)$  (Cross prod tells the dir. (for normal should be normalized) )

${E} = \vb{x}_4 - \vb{x}_3$ 

**Force**:

- Planar Case: $\mathbf{f}_{\mathbf{i}}=k \frac{\|\mathbf{E}\|^{2}}{\left\|\mathbf{N}_{1}\right\|+\left\|\mathbf{N}_{2}\right\|} \sin \left(\frac{\pi-\theta}{2}\right) \mathbf{u}_{i}$   (The magnitude of the cross product tells the area ($\|\vb{N}_i\|$))
- Non-planar Case: $\mathbf{f}_{\mathrm{i}}=k \frac{\|\mathbf{E}\|^{2}}{\left\|\mathbf{N}_{1}\right\|+\left\|\mathbf{N}_{2}\right\|}\left(\sin \left(\frac{\pi-\theta}{2}\right)-\sin \left(\frac{\pi-\theta_{0}}{2}\right)\right) \mathbf{u}_{i}$  (The initial angle is $\theta_0$ at rest, planar case $\theta_0 = 180^{\circ}$)

Explicit Derivative is difficult to complete / No energy

### A Quadratic Bending Model

2 assumptions: planar case (OK for cloth); little stretching (mainly bending)

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211207154104161.png" alt="image-20211207154104161" style="zoom:50%;" />

**Energy function**: (vector * matrix * vector^T^) ($\vb{I}$ is 3x3 identity)
$$
E(\mathbf{x})=\frac{1}{2}\left[\begin{array}{llll}
\mathbf{x}_{0} & \mathbf{x}_{1} & \mathbf{x}_{2} & \mathbf{x}_{3}
\end{array}\right] \mathbf{Q}\left[\begin{array}{l}
\mathbf{x}_{0} \\
\mathbf{x}_{1} \\
\mathbf{x}_{2} \\
\mathbf{x}_{3}
\end{array}\right];\quad
\vb{Q} = \frac{3}{A_0+A_1} \vb{qq}^{\mathrm{T}}\in \R^{12\times 12};\quad
\vb{q} = \begin{bmatrix}
\left(\cot \theta_{1}+\cot \theta_{3}\right) \vb{I} \\
\left(\cot \theta_{0}+\cot \theta_{2}\right) \vb{I} \\
\left(-\cot \theta_{0}-\cot \theta_{1}\right) \vb{I} \\
\left(-\cot \theta_{2}-\cot \theta_{3}\right) \vb{I} \end{bmatrix}\in \R ^{12\times3}\\
\Rightarrow E(\vb{x}) = \frac{3\|\vb{q}^{\mathrm{T}}\vb{x}\|^2}{2(A_0 + A_1)}; \text{ and } 
E(\vb{x}) = 0 \text{ when triangles are flat (co-planar)}
$$
Actually finding the laplacian (/ curl), when flat, no curl => $E(\vb{x}) = 0$ 

$\vb{Q}$ is a constant matrix => The function is quadratic 

**Pros & Cons**

- Easy to implement (even in implicit method)
  $$
  \vb{f(x)} = -\grad E(\vb{x}) = -\vb{Q} \begin{bmatrix}\vb{x}_0 \\ \vb{x}_1\\ \vb{x}_2 \\ \vb{x}_3\end{bmatrix}
  \quad \vb{H(x)} = \frac{\partial ^2 E(\vb{x})}{\partial \vb{x}^2} = \vb{Q}
  $$

- No longer valid if stretches much; Not suitable if the rest config is not planar -> (projective dynamics model / cubic shell model)

### The Locking Issue

In the mass-spring / other bending models, assuming *cloth planar deformation* and *cloth bending deformation* are **independent**

In zero bending case: LHS - OK; RHS - For **stiff spring** NO (Locking Issue) => short of **DoFs**. 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207161346501.png" alt="image-20211207161346501" style="zoom:50%;" />

For a manifold mesh, Euler formula: `#edges = 3#vertices - 3 - #boundary_edges`

If edges are all hard constraints: `DoFs = 3 + #boundary_edges`

=> no perfect solution

## Shape Matching

..



# Lecture 6 Constraint Approaches: PBD / PD / …

## Strain Limiting and Position Based Dynamics

### The Stiffness Issue

Real-world fabric *resist strongly* to stretching

But in simulation, only *increasing the stiffness* can cause: (More expensive / time-costing)

- *Explicit* integrators - **Unstable**

  -> *Smaller timesteps* and *more computational time*

- Linear systems invloved in *implicit* integrators will be **ill-conditioned**

  -> *More iter* and *computational time*

#### A Single Spring

If a spring is infinitely stiff -> Length = const

Def a constraint func: $\phi (\vb{x}) = \|\vb{x}_i - \vb{x}_j\| -L = 0$ (Rest length $L$) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207171930711.png" alt="image-20211207171930711" style="zoom:67%;" />

Def a proj func: suppose in a $\R^6$ space for a pos of $\vb{x}$ want to move into a rational area (in blue, with boundary $\delta \boldsymbol{\Omega}$) with a shortest path: want the projection

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207172245723.png" alt="image-20211207172245723" style="zoom: 67%;" />
$$
\left\{\mathbf{x}_{i}^{\text {new }}, \mathbf{x}_{j}^{\text {new }}\right\}=\operatorname{argmin} \frac{1}{2}\left\{m_{i}\left\|\mathbf{x}_{i}^{\text {new }}-\mathbf{x}_{i}\right\|^{2}+m_{j}\left\|\mathbf{x}_{j}^{\text {new }}-\mathbf{x}_{j}\right\|^{2}\right\}\\
\Downarrow\text{such that }\phi(\vb{x}) = 0\\
\left\{
\begin{aligned}
\vb{x}^{\text{new}}&\leftarrow \text{Projection}(\vb{x})\\
\mathbf{x}_{i}^{\text {new }} &\leftarrow \mathbf{x}_{i}-\frac{m_{j}}{m_{i}+m_{j}}\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|} \\
\mathbf{x}_{j}^{\text {new }} &\leftarrow \mathbf{x}_{j}+\frac{m_{i}}{m_{i}+m_{j}}\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|}
\end{aligned}\right.\\
\phi\left(\mathbf{x}^{\text {new }}\right)=\left\|\mathbf{x}_{i}^{\text {new }}-\mathbf{x}_{j}^{\text {new }}\right\|-L=\left\|\mathbf{x}_{i}-\mathbf{x}_{j}-\mathbf{x}_{i}+\mathbf{x}_{j}+L\right\|-L=0
$$
The 2 new points’ substraction equals the original length => satisfying the constraint

By default $m_i = m_j$, but can also set $m_i = \infty$ for stationary nodes (can be just ignored)

#### Multiple Springs

##### Gauss-Seidel Approach

Approach every spring sequentially in a certain order (needs a lot of iter to converge)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207173051472.png" alt="image-20211207173051472" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207173152018.png" alt="image-20211207173152018" style="zoom: 80%;" />

- Cannot ensure the satisafction of every constraint. More iter give much closer results (to the constraint)
- More similar to the *stochastic gradient descent* (in ML, with spec order)
- The order mattrers => cause *bias* and affect *convergence*

##### Jocabi Approach

Projects all edges **simultaneously** (good for *parallization*) and then **linearly blend** the results

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207173546431.png" alt="image-20211207173546431" style="zoom:80%;" />

sum up all and update together with a weighted average.

- Lower convergence rate
- More iter give better results

### Position Based Dynamics (PBD)

Based on the *proj func*, similar to shape matching

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207174356898.png" alt="image-20211207174356898" style="zoom:80%;" />

- $\vb{x}$ & $\vb{v}$ update as simple particle system (similar to Shape matching in rigid body collision)
- PBD: Add constraints
  - $\vb{x}$: use Gauss-Seidel / Jocobi /… to find the projection
  - $\vb{v}$: use new $\vb{x}$ to find 

**Properties**:

- The stiffness behavior is subject to non-physical factors
  - Iteration num (More iter, slower convergence, less stiffer)
  - Mesh resolution (Fewer vertices, faster convergence, stiffer)
- The velocity update following projection is important to dynamic effects
- Not only use in springs, but triangles, volumes, …

**Pros & Cons**:

> (Usually in 3D softwares)

- Pros:
  - Parallelable on GPUs (PhysX from NVIDIA)
  - Easy to implement
  - Fast in low res (Less than 1000 vertices)
  - Generic, can handle other coupling and constraints (including fluids)
- Cons:
  - Not physically correct (no acc sol, stiffness related to meshes account and iter num)
  - Low performance in high res
    - Hierarchical approaches (From low to high res. But causes oscillation and other issues) 
    - Acceleration approaches (Chebyshev, …)

### Strain Limiting

Some normal updates + strain limiting (corrections)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207203039808.png" alt="image-20211207203039808" style="zoom:80%;" />

e.g.: The constrain is not strictly equal to rest length $L$ but some constraints ($\sigma$ - stretching ratio as a limit)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207203146606.png" alt="image-20211207203146606" style="zoom: 50%;" />
$$
\vb{x}^{\text{new}} \leftarrow \text{Projection}(\vb{x})\\
\sigma \leftarrow \frac{1}{L}\|\vb{x}_i-\vb{x}_j \|\\
\sigma \leftarrow \min(\max (\sigma,\sigma^{\min}), \sigma^{\max})\quad \in[\sigma^{\min}, \sigma ^{\max}]\\
\left\{\begin{aligned}
\mathbf{x}_{i}^{\text {new }} &\leftarrow \mathbf{x}_{i}-\frac{m_{j}}{m_{i}+m_{j}}\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-\sigma_{0} L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|}\\
\mathbf{x}_{j}^{\text {new }} &\leftarrow \mathbf{x}_{j}-\frac{m_{j}}{m_{i}+m_{j}}\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|-\sigma_{0} L\right) \frac{\mathbf{x}_{i}-\mathbf{x}_{j}}{\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|}
\end{aligned}\right.\\
\text{PBD: }\sigma\equiv 1;\quad \text{No limits: }\sigma^{\min}, \sigma^{\max} \leftarrow \infty
$$
Can be used to simulate some cloth whose stiffness increases when being stretched heavily.

The constraints can be used for reducing oscillation for FEM / …

### Triangle Area Limit

Limit the triangle area. Define a scaling factor first $s = \sqrt{ \min(\max(A,A^{\min}),A^{\max})/A}$  (Mass center no change)
$$
\left\{\mathbf{x}_{i}^{\text {new }}, \mathbf{x}_{i}^{\text {new }}, \mathbf{x}_{k}^{\text {new }}\right\}=\operatorname{argmin}\frac{1}{2}\left\{m_{i}\left\|\mathbf{x}_{i}^{\text {new }}-\mathbf{x}_{i}\right\|^{2}+m_{j}\left\|\mathbf{x}_{j}^{\text {new }}-\mathbf{x}_{j}\right\|^{2}+m_{j}\left\|\mathbf{x}_{k}^{\text {new }}-\mathbf{x}_{k}\right\|^{2}\right\}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207204422565.png" alt="image-20211207204422565" style="zoom: 67%;" />

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207204545942.png" alt="image-20211207204545942" style="zoom:67%;" />

**Properties**:

- **Avoiding instability and artifacts** due to *large deformation*

- Useful for **nonlinear effects** (*Biphasic* way)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211207204745741.png" alt="image-20211207204745741" style="zoom: 50%;" /> 

- Help address the **locking issue**

## Projective Dynamics

### Main Idea

Instead of using projections to update $\vb{x}$ directly as in PBD, projective dynamics uses **projection** to define a **quadratic energy** (identical to spring energy, but have 2 intermediate results $\vb{x}_{e,i}^{\text{new}}$ & $\vb{x}_{e,j}^{\text{new}}$)
$$
E(\vb{x}) = \sum_{e=\{i,j\}} \frac{k}{2} \left\|(\vb{x}_i - \vb{x}_j) - (\vb{x}^{\text{new}}_{e,i} - \vb{x}^{\text{new}}_{e,j})\right\|^2 
=\sum_{e=\{i,j\}} \frac{k}{2}\left\| \|\vb{x}_i - \vb{x}_j\|\frac{\vb{x}_{i} - \vb{x}_{j}}{\|\vb{x}_{i} - \vb{x}_{j}\|} - L_e\frac{\vb{x}_{i} - \vb{x}_{j}}{\|\vb{x}_{i} - \vb{x}_{j}\|}\right\|^2 
= \sum_{e=\{i,j\}} \frac{k}{2} \left(\|\vb{x}_i - \vb{x}_j\| -L_e\right)^2
\\
\{\vb{x}^{\text{new}}_{e,i} - \vb{x}^{\text{new}}_{e,j}\} = \text{Projection}_e (\vb{x}_i,\vb{x}_j) \text{ for every edge } e
$$
Finding the corresponding **forces**: (assumption: $(\vb{x}_{e,i}^{\text{new}} - \vb{x}_{e,j}^{\text{new}}) = \text{const}$) (identical to spring force)
$$
\vb{f}_i = -\grad _iE(\vb{x}) = -k\sum_{e:\ i\in e} (\vb{x}_i - \vb{x}_j)  - (\vb{x}_{e,i}^{\text{new}} - \vb{x}_{e,j}^{\text{new}}) 
= -k\sum_{e:\ i\in e}\left(\|\vb{x}_i - \vb{x}_j\| -L_e\right)\frac{\vb{x}_{i} - \vb{x}_{j}}{\|\vb{x}_{i} - \vb{x}_{j}\|}
$$
=> althought the results are the same, the **Hessians** are different

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211214000539781.png" alt="image-20211214000539781" style="zoom: 67%;" />
$$
\begin{aligned}
E(\vb{x}) =& \frac{k}{2} \left\|(\vb{x}_0 - \vb{x}_1) - (\vb{x}^{\text{new}}_{0,0} - \vb{x}^{\text{new}}_{0,1})\right\|^2  + \frac{k}{2} \left\|(\vb{x}_0 - \vb{x}_2) - (\vb{x}^{\text{new}}_{1,0} - \vb{x}^{\text{new}}_{1,2})\right\|^2 + \frac{k}{2} \left\|(\vb{x}_0 - \vb{x}_3) - (\vb{x}^{\text{new}}_{2,0} - \vb{x}^{\text{new}}_{2,3})\right\|^2 +\\
& \frac{k}{2} \left\|(\vb{x}_1 - \vb{x}_2) - (\vb{x}^{\text{new}}_{3,1} - \vb{x}^{\text{new}}_{3,2})\right\|^2  + \frac{k}{2} \left\|(\vb{x}_2 - \vb{x}_3) - (\vb{x}^{\text{new}}_{4,2} - \vb{x}^{\text{new}}_{4,3})\right\|^2 \\

\end{aligned}
$$

$$
\vb{f}_0
 = \left[k\left(\mathbf{x}_{0,0}^{\text {new }}-\mathbf{x}_{0,1}^{\text {new }}\right)-k\left(\mathbf{x}_{0}-\mathbf{x}_{1}\right)\right]+\left[k\left(\mathbf{x}_{1,0}^{\text {new }}-\mathbf{x}_{1,2}^{\text {new }}\right)-k\left(\mathbf{x}_{0}-\mathbf{x}_{2}\right)\right]+  \left[k\left(\mathbf{x}_{2,0}^{\text {new }}-\mathbf{x}_{2,3}^{\text {new }}\right)-k\left(\mathbf{x}_{0}-\mathbf{x}_{3}\right)\right]
$$

$$
\vb{H} = \begin{bmatrix}
3k\vb{I} & -k\vb{I} & -k\vb{I} & -k\vb{I}\\
-k\vb{I} & 2k\vb{I} & -k\vb{I} & \\
-k\vb{I} & -k\vb{I} & 3k\vb{I} & -k\vb{I}\\
-k\vb{I} & & -k\vb{I} & 2k\vb{I}
\end{bmatrix}
$$

The Hessian is the 2nd order derivative of energies to the vertices. The coefficients (of the diagonal) can be regarded as the numbers of the springs the vertices connected (3, 2, 3, 3 for vertex 0-3) 

This matrix is simple and constant $\vb{A = }\frac{1}{\Delta t^2} \vb{M+H}$

=> Applying implicit integration

### Simulation by PD

Combination fo the projectiive dynamics with implicit time integration. Use a direct LU solver and can factorize $\vb{A}$ once (usually the most expensive part is the factorization and here the costs can reduce) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211214002319156.png" alt="image-20211214002319156" style="zoom:67%;" />

### Principles

Newton’s Method calculate $\vb{H}$ as Hessian
$$
\underbrace{\left(\frac{1}{\Delta t^2}\vb{M + H}\right)}_{\text{Hessian}} \Delta \vb{x} =  \underbrace{-\frac{1}{\Delta t^2} \vb{M (\vb{x}^{(k)}-\vb{x}^{[0]} - \Delta t\vb{v^{[0]}}) + \vb{f}(\vb{x}^{(k)})} }_{\text{Negative gradient}}
$$
But in practice, $\vb{H}$ usually not calculated exactly (The performance dep. on how well the Hessian is approx., identity matrix can even be used (but convergence takes longer))

- Truncating the Hessian to make it p.d. (Lec. 5)
- A (magical) diagonal approx (Lab 2)
- A const matrix by PD (Lec. 6)

### Pros and Cons

**Pros**

- Have theoretical solution with physical meaning
- Fast on CPUs with a direct solver (one factorization)
- Fast convergence in the first few iter

**Cons**

- Slow on GPUs 
- Slow convergence over time (fails to consider Hessian caused by projection) (suffering from high stiffness)
- Cannot handle constraint changes easily (contacts / remeshing due to fracture / …)

## Constrained Dynamics

This method is suitable for **strong constraints / contacts / articulated rigid bodies (ragdoll animation) …**

> A critical problem: what if constraints/forces are very stiff (or even infinitely stiff)
>
> example: multiple rigid bodies: human body can be regarded as several rigid bodies with very strong constraints (joints)

Compliant constraint: $\phi_e(\vb{x}) = \|\vb{x}_{ei}-\vb{x}_{ej}\| - L_e$  

The energy can be rewrited:
$$
E(\vb{x}) = \sum_e \frac{1}{2} k {\underbrace{(\| \vb{x}_{ei} - \vb{x}_{ej} -L_e \|)}_{\phi_e}} ^2 = \frac{1}{2}\vb{\Phi}^{\mathrm{T}}(\vb{x}) \vb{C}^{-1} \vb{\Phi}(\vb{x})\\
\vb{f(x)} = -\grad E = -\left( \frac{\partial E}{\partial \vb{\Phi}}\frac{\partial \vb{\Phi}}{\partial \vb{x}}\right)^{\mathrm{T}} = -\vb{J}^{\mathrm{T}}\vb{C}^{-1}\vb{\Phi} = \vb{J}^{\mathrm{T}}\vb{\lambda}
$$
Let $N$ be the number of vertices and $E$ be the number of constraints ($\vb{C}$ - compliant matrix, $\vb{J}$ - Jacobian, $\vb{\lambda}$ - Lagrangian multipliers (Dual variables))
$$
\vb\Phi(\vb{x})\in \R^{E} \ , \begin{bmatrix}\phi_0\\ \phi_1 \\ \vdots \\ \phi_E\end{bmatrix}\ ;\quad
\vb{C} = \begin{bmatrix}\frac{1}{k} & & \\ & \frac{1}{k } & \\ & & \ddots\end{bmatrix}\in \R^{E\times E}\ ;\quad
\vb{J } = \frac{\partial \vb{\Phi}}{\partial \vb{x}}  \in \R^{E\times 3N}\ ;\quad
\vb{\lambda} = -\vb{C} ^{-1} \vb{\Phi}\in \R^{E}
$$
By implicit Integration: (2 sets of var: primal var $\vb{x}$ (or $\vb{v = \dot {x}}$) and the dual var $\lambda$)
$$
\left\{\begin{aligned}
&\vb{Mv}^{\text{new}} - \Delta t \vb{J}^{\mathrm{T}}\vb{\lambda}^{\text{new}} = \vb{Mv}\\
&\vb{C\lambda} ^{\text{new}} = -\vb{\Phi}^{\text{new}} \approx - \vb{\Phi} - \vb{J}(\vb{x}^{\text{new}} - \vb{x}) \approx -\vb{\Phi} - \Delta t\ \vb{J} \vb{v}^{\text{new}}
\end{aligned}\right.\\
\Rightarrow 
\begin{bmatrix}\vb{M} & -\Delta t \vb{J}^{\mathrm{T}} \\ \Delta t\vb{J} & \vb{C} \end{bmatrix}
\begin{bmatrix}\vb{v}^{\text{new}} \\ \vb{\lambda}^{\text{new}} \end{bmatrix} = 
\begin{bmatrix}\vb{Mv} \\ - \vb{\Phi} \end{bmatrix}
$$

- Method 1: Solve by a direct solver together (**primal-dual fashion**): The matrix should be p.d. => need to make p.d.

- Method 2: Reduce the system by Schur complement and solve $\vb{\lambda}^{\text{new}}$ first (not easy to reduce / not always sparse)
  $$
  \begin{gathered}
  \left(\Delta t^{2} \mathbf{J M}^{-1} \mathbf{J}^{\mathrm{T}}+\mathbf{C}\right) \lambda^{\text {new }}=-\boldsymbol{\Phi}-\Delta t \mathbf{v} \\
  \mathbf{v}^{\text {new }} \leftarrow \mathbf{v}+-\Delta t \mathbf{M}^{-1} \mathbf{J}^{\mathrm{T}} \boldsymbol{\lambda}^{\text {new }}
  \end{gathered}
  $$
  If reduce the $\lambda$ first -> get the implicit method formula

Infinite stiffness in this case could be solved by $\vb{C}\rightarrow 0$ (smaller $\vb{C}$ makes the system better)  



# Lecture 7 Linear Finite Element Method (FEM)

## Linear Finite Element Method (FEM)

### The Linear FEM Assumption

In a nutshell, linear FEM assumes that for any point $\vb{X} $ in the reference triangles, its deformed correspondence is $\vb{x = FX + c}$ (“Linear” for this linear formula)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211214162153805.png" alt="image-20211214162153805" style="zoom:80%;" />

For any vectors between 2 points we can use $\vb{F}$ to convert it from ref to deformed state:
$$
\vb{x}_{ba} = \vb{x}_b - \vb{x}_a = \vb{FX}_b +\vb{c} - \vb{FX}_a-\vb{c} = \vb{FX}_{ba}\\
$$
For example for edge 10 & 20: (indep on translation)
$$
\vb{FX}_{10} = \vb{x}_{10}\ ; \ \vb{FX}_{20} = \vb{x}_{20} \Rightarrow \vb{F} = \begin{bmatrix}\vb{x}_{10} & \vb{x}_{20} \end{bmatrix} \begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} \end{bmatrix}^{-1}
$$
Problem: $\vb{F}$ is related to deformation but contains rotations (want to also reduce rotation)

### Green Strain

Ideally, we need a tensor to describe shape deformation only. Recall that **SVD** gives $\vb{F=UDV}^{\mathrm{T}}$, where only $\vb{V}^{\mathrm{T}}$ and $\vb{D}$ are relevant to deformation.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211109165409060.png" alt="image-20211109165409060" style="zoom:70%;" />

Get rid of $\vb{U}$ as: $\vb{G}$ (**Green Strain**, symmetric, only 3 var)  ($\vb{F}^{\mathrm{T}}\vb{F} = \vb{VD}\cancel{\vb{U}^{\mathrm{T}}\vb{U}}\vb{DV}^{\mathrm{T}} $) 
$$
\vb{G} = \frac{1}{2} (\vb{F}^{\mathrm{T}}\vb{F} - \vb{I}) = \frac{1}{2} (\vb{VD}^{2}\vb{V}^{\mathrm{T}} - \vb{I}) = \begin{bmatrix}\varepsilon_{uu}& \varepsilon_{uv}\\\varepsilon_{uv} & \varepsilon_{vv} \end{bmatrix}
$$

- If no deformation: $\vb{G=0}$; if deformation increases, $\|\vb{G}\|$ increases
- 3 deformation modes: $\varepsilon_{uu},\varepsilon_{uv}, \varepsilon_{vv}$ 
- $\vb{G}$ is rotaion irrelavent (if adding rotaion $\vb{R}$ then deformation grad is $\vb{RF}$ but $\vb{G}$ is the same)  

### Strain Energy Density Function

Consider the energy density per ref area: $W(\vb{G})$:

**Total energy**:
$$
E=\int W(\vb{G})\ \mathrm{d}A = A^{\text{ref}}W(\varepsilon_{uu},\varepsilon_{uv}, \varepsilon_{vv})
$$
The **Saint Venant-Kirchhoff Model (StVK)**: ($\lambda$ and $\mu$ are Lamé parameters)
$$
W\left(\varepsilon_{u u}, \varepsilon_{v v}, \varepsilon_{u v}\right)=\frac{1}{2}\left(\varepsilon_{u u}+\varepsilon_{v v}\right)^{2}+\mu\left(\varepsilon_{u u}^{2}+\varepsilon_{v v}^{2}+2 \varepsilon_{u v}^{2}\right)
$$

$$
\frac{\partial W}{\partial \mathbf{G}}=\left[\begin{array}{cc}
\frac{\partial W}{\partial \varepsilon_{u u}} & \frac{1}{2} \frac{\partial W}{\partial \varepsilon_{u v}} \\
\frac{1}{2} \frac{\partial W}{\partial \varepsilon_{u v}} & \frac{\partial W}{\partial \varepsilon_{v v}}
\end{array}\right]=\left[\begin{array}{cc}
2 \mu \varepsilon_{u u}+\lambda \varepsilon_{u u} & 2 \mu \varepsilon_{u v} \\
2 \mu \varepsilon_{u v} & 2 \mu \varepsilon_{v v}+\lambda \varepsilon_{v v}
\end{array}\right]=2 \mu \mathbf{G}+\lambda \operatorname{trace}(\mathbf{G}) \mathbf{I}=\mathbf{S}
$$

(Trace is the diagonal sum; $\vb{S}$ - Second Piola-Kirchhoff stress tensor (some form of force density))

**Force**:
$$
\mathbf{f}_{i}=-\left(\frac{\partial E}{\partial \mathbf{x}_{i}}\right)^{\mathrm{T}}=-A^{\mathrm{ref}}\left(\frac{\partial W}{\partial \mathbf{x}_{i}}\right)^{\mathrm{T}}=-A^{\mathrm{ref}}\left(\frac{\partial W}{\partial \varepsilon_{u u}} \frac{\partial \varepsilon_{u u}}{\partial \mathbf{x}_{i}}+\frac{\partial W}{\partial \varepsilon_{v v}} \frac{\partial \varepsilon_{v v}}{\partial \mathbf{x}_{i}}+\frac{\partial W}{\partial \varepsilon_{u v}} \frac{\partial \varepsilon_{u v}}{\partial \mathbf{x}_{i}}\right)^{\mathrm{T}}
$$
The $\frac{\partial W}{\partial \varepsilon_{ij}}$ are solved before, we concentrate more on the next part

Recall that: $\mathbf{F}=\left[\begin{array}{ll}
\mathbf{x}_{10} & \mathbf{x}_{20}
\end{array}\right]\left[\begin{array}{ll}
\mathbf{r}_{10} & \mathbf{r}_{20}
\end{array}\right]^{-1}=\left[\begin{array}{ll}
\mathbf{x}_{10} & \mathbf{x}_{20}
\end{array}\right]\left[\begin{array}{ll}
a & b \\
c & d
\end{array}\right]=\left[\begin{array}{ll}
a \mathbf{x}_{10}+c \mathbf{x}_{20} & b \mathbf{x}_{10}+d \mathbf{x}_{20}
\end{array}\right]$ 

By def
$$
\mathbf{G}=\frac{1}{2}\left(\mathbf{F}^{\mathrm{T}} \mathbf{F}-\mathbf{I}\right)=\left[\begin{array}{cc}
\frac{1}{2}\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)-\frac{1}{2} & \frac{1}{2}\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right) \\
\frac{1}{2}\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right) & \frac{1}{2}\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}}\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)-\frac{1}{2}
\end{array}\right]
$$

$$
\Rightarrow
\left\{\begin{aligned}
&\frac{\partial \varepsilon_{u u}}{\partial \mathbf{x}_{1}}=a\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}} \quad \frac{\partial \varepsilon_{v v}}{\partial \mathbf{x}_{1}}=b\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}} \quad \frac{\partial \varepsilon_{u v}}{\partial \mathbf{x}_{1}}=\frac{1}{2} a\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}}+\frac{1}{2} b\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}\\
&\frac{\partial \varepsilon_{u u}}{\partial \mathbf{x}_{2}}=c\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}} \quad \frac{\partial \varepsilon_{v v}}{\partial \mathbf{x}_{2}}=d\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}} \quad \frac{\partial \varepsilon_{u v}}{\partial \mathbf{x}_{2}}=\frac{1}{2} c\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}}+\frac{1}{2} d\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}
\end{aligned} \right.
$$

(the method is too complex => simplification)
$$
\begin{aligned}
\mathbf{f}_{1}&=-A^{\mathrm{ref}}\left(\frac{\partial W}{\partial \varepsilon_{u u}} \frac{\partial \varepsilon_{u u}}{\partial \mathbf{x}_{1}}+\frac{\partial W}{\partial \varepsilon_{v v}} \frac{\partial \varepsilon_{v v}}{\partial \mathbf{x}_{1}}+\frac{\partial W}{\partial \varepsilon_{u v}} \frac{\partial \varepsilon_{u v}}{\partial \mathbf{x}_{1}}\right)^{\mathrm{T}}\\
&=-A^{\mathrm{ref}}\left(\frac{\partial W}{\partial \varepsilon_{u u}} a\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}+\frac{\partial W}{\partial \varepsilon_{v v}} b\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}}+\frac{\partial W}{\partial \varepsilon_{u v}} \frac{1}{2} a\left(b \mathbf{x}_{10}+d \mathbf{x}_{20}\right)^{\mathrm{T}}+\frac{\partial W}{\partial \varepsilon_{u v}} \frac{1}{2} b\left(a \mathbf{x}_{10}+c \mathbf{x}_{20}\right)^{\mathrm{T}}\right)\\
&=-A^{\mathrm{ref}}\left(\left[\begin{array}{cc}
a \mathbf{x}_{10}+c \mathbf{x}_{20} & b \mathbf{x}_{10}+d \mathbf{x}_{20}
\end{array}\right]\left[\begin{array}{cc}
\frac{\partial W}{\partial \varepsilon_{u u}} a+\frac{\partial W}{\partial \varepsilon_{u v}} \frac{1}{2} b \\
\frac{\partial W}{\partial \varepsilon_{u v}} \frac{1}{2} a+\frac{\partial W}{\partial \varepsilon_{v v}} b
\end{array}\right]\right)\\
&=-A^{\operatorname{ref}}
\underbrace{\left[a \mathbf{x}_{10}+c \mathbf{x}_{20} \quad b \mathbf{x}_{10}+d \mathbf{x}_{20}\right]}_{\text{deformation gradient}}
\underbrace{\left[\begin{array}{cc}
\frac{\partial W}{\partial \varepsilon_{u u}} & \frac{1}{2} \frac{\partial W}{\partial \varepsilon_{u v}} \\
\frac{1}{2} \frac{\partial W}{\partial \varepsilon_{u v}} & \frac{\partial W}{\partial \varepsilon_{v v}}
\end{array}\right]}_{\text{Second Piola-Kirchhoff Stress}}
\left[\begin{array}{l}
a \\
b
\end{array}\right]=-A^{\text{ref}} \vb{FS}\begin{bmatrix}a \\ b\end{bmatrix}
\end{aligned}
$$

$$
\Rightarrow [\vb{f}_1\ \ \vb{f}_2] = -A^{\text{ref}}\ \vb{FS}\ [\vb{r}_{10}\ \ \vb{r}_{20}]^{-1}\ ;\quad \vb{f}_0 = -\vb{f}_1 - \vb{f}_2
$$

For tetrahedron (3D ref -> 3D deformation)

## Finite Volume Method (FVM)

> For simple triangle / tetrahedra, FVM is equivalent to linear FEM

FVM considers force calculation in an **integration perspective** other than a differentiation perspective

### Traction and Stress in FVM

#### Basic Ideas in FVM

**General Case**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221120056804.png" style="zoom:67%;" />

- **Traction** $\vb{t}$: The internal force per unit length (area) => the **total force at the interface** is the integral: $\vb{f } = \oint_L \vb{t}\ \mathrm{d}l$ 
- **Stress tensor $\vb{\sigma}$**: $\vb{t = \sigma n}$ (projection: interface normal -> traction)

**The Triangle Case**: Take the midpoints which will be on the curve to estimate (the force contributes the same to the 3 vertices, the integral is for $\vb{x}_0$) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221120134281.png" alt="image-20211221120134281" style="zoom: 50%;" />

- **Force contributed by an element**: $\vb{f}_0 = \oint _L \vb{\sigma n }\ \mathrm{d}l$ (the integral of traction)
- **Divergence theorem**: $\oint_L \vb{\sigma n}\ \mathrm{d}l + \oint_{L_{20}} \vb{\sigma n}\ \mathrm{d}l + \oint_{L_{10}}\vb{\sigma n}\ \mathrm{d}l = 0$ ($\vb{\sigma}$ is constant within the elem)
- **Force**: $\vb{f}_0 = -\oint_{L_{20}}\vb{\sigma n}_{20}\ \mathrm{d}l -\oint_{L_{10}}\vb{\sigma n}_{10}\ \mathrm{d}l = -\vb{\sigma} (\frac{\|\vb{x}_{20}\|}{2}\vb{n}_{20} + \frac{\|\vb{x}_{10}\|}{2}\vb{n}_{10})  $  (Only $\sigma$ is unknown)

**In 3D (tetrahedra)**: FVM does the surfacial integral

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221170657979.png" alt="image-20211221170657979" style="zoom: 67%;" />

- **Force**: 
  $$
  \begin{aligned}
  \mathbf{f}_{0} &=-\oint_{\Omega}^{\sigma} \boldsymbol{n} d A=-\boldsymbol{\sigma}\left(\frac{A_{012}}{3} \mathbf{n}_{012}+\frac{A_{023}}{3} \mathbf{n}_{023}+\frac{A_{031}}{3} \mathbf{n}_{031}\right) \\
  &=-\frac{\boldsymbol{\sigma}}{3}\left(\frac{\left\|\mathbf{x}_{10} \times \mathbf{x}_{20}\right\|}{2} \frac{\mathbf{x}_{10} \times \mathbf{x}_{20}}{\left\|\mathbf{x}_{10} \times \mathbf{x}_{20}\right\|}+\frac{\left\|\mathbf{x}_{20} \times \mathbf{x}_{30}\right\|}{2} \frac{\mathbf{x}_{20} \times \mathbf{x}_{30}}{\left\|\mathbf{x}_{20} \times \mathbf{x}_{30}\right\|}\right.
  \left.+\frac{\left\|\mathbf{x}_{30} \times \mathbf{x}_{10}\right\|}{2} \frac{\mathbf{x}_{30} \times \mathbf{x}_{10}}{\left\|\mathbf{x}_{30} \times \mathbf{x}_{10}\right\|}\right) \\
  &=-\frac{\boldsymbol{\sigma}}{6}\left(\mathbf{x}_{10} \times \mathbf{x}_{20}+\mathbf{x}_{20} \times \mathbf{x}_{30}+\mathbf{x}_{30} \times \mathbf{x}_{10}\right)
  \end{aligned}
  $$

#### Different Stresses

The **stress tensor** here: **mapping** from the **interface normal** to the **traction** (but kind of different from FEM)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221171303566.png" alt="image-20211221171303566" style="zoom:67%;" />

- In FEM: Stress $\vb{S}$, normal $\vb{N}$ and traction $\vb{T}$ are in the **reference** state
- In FVM: this stress assumes the normal $\vb{n}$ and the traction $\vb{t}$ are in the **deformed** state 

| Output \ Input                             | Interface normal $\vb{N}$ in the *ref* state (Undeformed) | Interface normal $\vb{n}$ in the *current* state (Deformed)  |
| ------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------ |
| **Traction in the *ref* state (unformed)** | Second Piola-Kirchhoff stress ($\vb{S}$)                  | -                                                            |
| **Traction in the *cur* state (formed)**   | First Piola-Kirchhoff stress ($\vb{P}\vb{= FS}$)          | Cauchy Stress ($\vb{\sigma} = \det ^{-1} \vb{(F)PF}^{\mathrm{T}} = \det^{-1}\vb{(F)FSF^{\mathrm{T}}}$) |

($\vb{F}$ - Deformation gradient) 

#### The Finite Volume Method

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221214405998.png" alt="image-20211221214405998" style="zoom: 67%;" />
$$
\begin{aligned}
\vb{f}_1 &= -\frac{\sigma}{6} (\vb{x}_{01}\times \vb{x}_{21} + \vb{x}_{21} \times \vb{x}_{31} + \vb{x}_{31}\times \vb{x}_{01}) \qquad \ \ \ (\text{Replace $\sigma$  with }\vb{P}\text{ (=> normal)} )
\\
&= -\frac{\vb{P}}{6} (\vb{X}_{01} \times \vb{X}_{21} + \vb{X}_{21} \times \vb{X}_{31} + \vb{X}_{31} \times \vb{X}_{01}) \quad (\text{The term including } \vb{X} \text{ is constant => precomputed})
\\ 
&= -\frac{\vb{FS}}{6} \vb{b}_1\qquad (\vb{b}_1 \text{ is the constant}) 
\end{aligned}
$$
Second Piola-Kirchhoff Stress: $\vb{S} = \frac{\partial W}{\partial \vb{G}}$ as in previous FEM formulation

=> To find $\vb{b}_1$ (In the figure below, $\vb{X}_1$ will be computed via $\vb{b}_1$) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221220647936.png" alt="image-20211221220647936" style="zoom:67%;" />

The dot product of the edge 10 with $\vb{X}_1$ 
$$
\begin{aligned}
\mathbf{X}_{10}^{\mathrm{T}} \mathbf{b}_{1} &=\mathbf{X}_{10}^{\mathrm{T}}\left(\cancel{\mathbf{X}_{01} \times \mathbf{X}_{21}}+\mathbf{X}_{21} \times \mathbf{X}_{31}+\cancel{\mathbf{X}_{31} \times \mathbf{X}_{01}}\right)=\mathbf{X}_{10}^{\mathrm{T}}\left(\mathbf{X}_{21} \times \mathbf{X}_{31}\right) \\
&=\mathbf{X}_{01}^{\mathrm{T}}\left(\mathbf{X}_{31} \times \mathbf{X}_{21}\right)=6\
\mathrm{Vol}
\end{aligned}
$$
Meanwhile: $\vb{X}_{20}^{\mathrm{T}}$ & $\vb{X}_{30}^{\mathrm{T}}$ equal to 0
$$
\Rightarrow \begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{\mathrm{T}} \vb{b}_1 = \begin{bmatrix} 6\ \mathrm{Vol} \\ 0 \\  0 \end{bmatrix}\ ;
\quad
\begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{\mathrm{T}} \vb{b}_2 = \begin{bmatrix}  0 \\ 6\ \mathrm{Vol} \\ 0 \end{bmatrix}\ ;
\quad
\begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{\mathrm{T}} \vb{b}_3 = \begin{bmatrix}  0 \\ 0 \\ 6\ \mathrm{Vol}  \end{bmatrix}\\
\Rightarrow \begin{bmatrix}\vb{b}_{1} & \vb{b}_{2} & \vb{b}_{3} \end{bmatrix} = 6\ \mathrm{Vol}\ \begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{-\mathrm{T}} 
 = \frac{1}{\det(\begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{-1})} \begin{bmatrix}\vb{X}_{10} & \vb{X}_{20} & \vb{X}_{30} \end{bmatrix}^{-\mathrm{T}}
$$

#### Algorithm (Explicit) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211221225719287.png" alt="image-20211221225719287" style="zoom: 67%;" />

-> Various material (Heterogeneous materials) / various elastic / …

## Hyperelastic Models

> **StVK**: cannot handle flips / … (only widely used in graphics)
>
> **Hyperelastic**: More used in engineering (material / mechanics /… ) => more general

### Isotropic Materials

#### Isotropic

Force on any direction affects the same (with same behaviors) (-> ansotropic)

$\vb{F}$ - Green Strain -> SVD; $\vb{D} = \lambda_0, \lambda_ 1, \lambda_2$ - **Principal stretches**: the singlar value of $\vb{F}$ 
$$
\vb{P(F)} = \vb{P(UDV}^{\mathrm{T}}) = \vb{UP} (\lambda_0 , \lambda_1, \lambda_2) \vb{V}^{\mathrm{T}}
$$

In many lit, parameterize $\vb{P}(I_{\vb{C}},II_{\vb{C}}, III_{\vb{C}})$ by principal invariants: ($\vb{C} = \vb{F}^{\mathrm{F}}\vb{U}$ is the right Cauchy-Green deformation tensor)
$$
I _{\vb{C}} = \trace (\vb{C}) = \lambda_0^2 + \lambda_1^2 + \lambda_2^2 \ ;\quad III_{\vb{C}} = \trace (\vb{C}^2)  = \lambda_0^4 +  \lambda_1^4 + \lambda_2^4 \\
II_\vb{C} = \frac{1}{2} \left(\trace^2(\vb{C}) - \tr(\vb{C}^2)\right) =  \lambda_0^2 \lambda_1^2 +   \lambda_0^2 \lambda_2^2 + \lambda_1^2 \lambda_2^2
$$

#### Models

- The **Saint Venant-Kirchhoff** model (StVK):
  $$
  W=\frac{s_{0}}{2}\left(I_{\mathbf{C}}-3\right)^{2}+\frac{s_{1}}{4}\left(I I_{\mathbf{C}}-2 I_{\mathbf{C}}+3\right)
  $$

- The **neo-Hookean** model: (More standard model in material mechanics)
  $$
  W=s_{0}\left(I I I_{\mathbf{C}}^{-1 / 3} I_{\mathbf{C}}-3\right)+s_{1}\left(I I I_{\mathbf{C}}^{-1 / 2}-1\right)
  $$
  Two terms of $W$: left - **Against shearing**; right - **Against bulky change** (volume change) (same as StVK)

- The **Mooney Rivlin** model:
  $$
  W=s_{0}\left(I I I_{\mathbf{C}}^{-1 / 3} I_{\mathbf{C}}-3\right)+s_{1}\left(I I I_{\mathbf{C}}^{-1 / 2}-1\right)+s_{2}\left(\frac{1}{2} I I I_{\mathbf{C}}^{-2 / 3}\left(I_{\mathbf{C}}^{2}-I I_{\mathbf{C}}\right)-3\right)
  $$

- The **Fung** model:
  $$
  W=s_{0}\left(I I I_{\mathbf{C}}^{-1 / 3} I_{\mathbf{C}}-3\right)+s_{1}\left(I I I_{\mathbf{C}}^{-1 / 2}-1\right)+s_{2}\left(e^{s_{3}\left(I I I_{\mathbf{C}}^{-1 / 3} I_{\mathbf{C}}-3\right)}-1\right)
  $$

#### Solving for First PK Stress ($\vb{P}$) 

Use the principal stretch for computation => first P-K stress
$$
\vb{P}(\lambda_0,\lambda_1,\lambda_2) = \begin{bmatrix} 
 \frac{\partial W}{\partial \lambda_0} & & \\
 & \frac{\partial W}{\partial \lambda_1} & \\
 & & \frac{\partial W}{\partial \lambda_2} 
\end{bmatrix}\quad
\Rightarrow 
\vb{P} = \vb{UP}(\lambda_0,\lambda_1,\lambda_2) \vb{V}^{\mathrm{T}} 
$$

### Algorithm

Note: for the same model the first PK in this algorithm should equiv to $\vb{P} = \vb{F}\frac{\partial W}{\partial \vb{G}}$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222113045756.png" alt="image-20211222113045756" style="zoom:67%;" />

### The Limitation of StVK

The compression (with high stress) and flip cannot be handled with StVK model (even get stable in the flipped side) => UNSTABLE

=> Neo Hookean solve the problem

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222113516604.png" alt="image-20211222113516604" style="zoom:67%;" />


# Lecture 8 Linear Finite Element Method II

## Nonlinear Optimization

### Gradient Descent

Solving $\vb{x}^* = \operatorname{argmin} F(\vb{x})$ => The negative gradient dir is the fastest descent dir

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222114523262.png" alt="image-20211222114523262" style="zoom:50%;" />

#### Algorithm

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222114657200.png" alt="image-20211222114657200" style="zoom:67%;" />

#### Step Size Adjustment

Linear search methods: Exact line search (solve another optimization problem) & Backtracking line search ($\beta$ is a smaller num (~ 0.3-0.4) -> Wolfe Condition)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222114751055.png" alt="image-20211222114751055" style="zoom:67%;" />

#### Descent Directions

The dir $\vb{d(x)}$ is descending if a sufficiently small step size $\alpha$ exists for: $F(\vb{x}) > F(\vb{x+ \alpha d(x)} )$ (at least descending)

ie. $-\grad F(\vb{x})\cdot \vb{d(x)}>0$ (the same side with the negative dir)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222211524599.png" alt="image-20211222211524599" style="zoom:67%;" />

With line search. use any search dir as long as it’s descending $F(\vb{x}^{(0)}) > F(\vb{x}^{(1)}) > F(\vb{x}^{(2)}) > F(\vb{x}^{(3)}) > \dots$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222211906481.png" alt="image-20211222211906481" style="zoom:67%;" />

#### Descent Methods

- **Gradient descent**: $\vb{d(x)} = -\grad F(\vb{x}) \Rightarrow -\grad F(\vb{x}) \cdot (-\grad F(\vb{x})) >0$  

- **Newton’s method** (if the Hessian is always positive definite): 
  $$
  \vb{d(x)} = -\left( \frac{\partial ^2 F(\vb{x})}{\partial \vb{x}^2}\right)^{-1} \grad F(\vb{x}) \Rightarrow -\grad F(\mathbf{x}) \cdot\left(-\left(\frac{\partial^{2} F(\mathbf{x})}{\partial \mathbf{x}^{2}}\right)^{-1} \grad F(\mathbf{x})\right)>0
  $$

- Any method using a <u>positive definite</u> matrix $\vb{P}$ to modify the gradient yields a descent method:
  $$
  \vb{d(x)} = - \vb{P}^{-1} \grad F(\vb{x}) \Rightarrow -\grad F(\vb{x}) \cdot \left(-\vb{P}^{-1} \grad F(\vb{x}) \right)>0
  $$

A unified descent framework:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222212628550.png" alt="image-20211222212628550" style="zoom:67%;" />

Differences between these methods:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222212648030.png" alt="image-20211222212648030" style="zoom:67%;" />

**Total cost = Per-iter cost * Num of iter**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211222212838495.png" alt="image-20211222212838495" style="zoom:50%;" />



# Lecture 9 Collision Handling

## Collision Detection

### Collision Detection Pipeline

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228113838360.png" alt="image-20211228113838360" style="zoom:67%;" />

- **Broad-Phase**: Remove the pairs which are impossible to collide
  - SH: Easy to implement; GPU friendly; needs to recompute after updating obj
  - BVH: More involved; not GPU friendly; update BV to update BVH

- **Narrow-Phase**: Which pairs really collide

### Board-Phase Collision Culling

#### Spatial Hashing (Partitioning)

> Static space division

Spatial partitioning divides the space by a grid and stores objects into grid cells

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228114157341.png" alt="image-20211228114157341" style="zoom:50%;" />

| Cell 12: $t_4$       | Cell 13: $t_4$       | Cell 14: $t_5$         | Cell 15: $t_5$ |
| -------------------- | -------------------- | ---------------------- | -------------- |
| Cell 8: $t_2$        | Cell 9: $t_3$        | Cell 10: $t_3$, $ t_5$ | Cell 11: N/A   |
| Cell 4: $t_0$, $t_2$ | Cell 5: $t_0$, $t_3$ | Cell 6: $t_3$          | Cell 7: $t_1$  |
| Cell 0: $t_0$        | Cell 1: N/A          | Cell 2: $t_1$          | Cell 3: $t_1$  |

For example for $t_3$: Cell 5, 6, 9, 10 => other triangles stored in these cells: $t_0$ & $t_5$ => pairs: $t_0t_3$ & $t_3t_5$ and do further detection

For moving objects, just expand the object region (add more)

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211228114759191.png" alt="image-20211228114759191" style="zoom:67%;" />

**Optimization** for memory costs: Memory wastes in empty cells and too many meshes should be built in 3D. Instead of allocating memories to cell, build an **object-cell list** and **sort** them

e.g., (not relavent to space)

```
{0, t_0}; {4, t_0}; {5, t_0}; {2, t_1}; {3, t_1}; {7, t_1}; {4, t_2}; {8, t_2}; {5, t_3}; {6, t_3}; {9, t_3}; {10, t_3}; {12, t_4};  {13, t_4}; {10, t_5}; {14, t_5}; {15, t_5} 
```

Then sort by cell ID => easy to find out what is stored in each cell

```
{0, t_0}; {2, t_1}; {3, t_1}; {4, t_0}; {4, t_2}; {5, t_0}; {5, t_3}; {6, t_3}; {7, t_1}; {8, t_2}; {9, t_3}; {10, t_3}; {10, t_5}; {12, t_4};  {13, t_4}; {14, t_5}; {15, t_5} 
```

**Morton Code**

One question is how to define the cell ID. Using the grid order is not optimal, since it cannot be easily extended and it is lack of locality (LHS). Morton code uses a Z-pattern instead (RHS).

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211228164349900.png" alt="image-20211228164349900" style="zoom:67%;" />

#### Bounding Volume Hierarchy (BVH)

Bounding volume hierarchy is built on **geometric/topological proximity** of objects.

##### External Object

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228170551965.png" alt="image-20211228170551965" style="zoom:67%;" />

The parts that are not like to collide in different boxes => test if box intersection occurs

To find elem potentially in collision with an obj, just traverse the tree:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228171141111.png" alt="image-20211228171141111" style="zoom:67%;" />

##### Internal Intersections (Self Collision)

2 procedures in the tree

``` c
Process_Node(A)
{	// check if the parental node (A) has self-collision
 	For every A’s child: B
        Process_Node(B)
    For every A’s children pair <B, C>
        if B and C intersect
            Process_Pair(B, C)	// the next precedure
}
```

``` c
Process_Pair(B, C)
{	// check if the children pair intersects
    For every B’s child: B’
    For every C’s child: C’
        if B’ and C’ intersect
            Process_Pair(B’, C’)
}
```

##### BVH Models

<u>Axis-aligned bounding box (AABB)</u> / Spheres / Oriented bounding box (OBB)

Two AABBs intersect if and only if they intersect in every axis; shperes: need to compute the distance

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228172043462.png" alt="image-20211228172043462" style="zoom: 50%;" />

### Narrow-Phase Collision Test

#### Discrete Collision Detection (DCD)

> Actually not collision detection but intersection detection

DCD tests if any intersection exists in each state at <u>discrete time instant</u>: $\vb{x}^{[0]}, \vb{x}^{[1]}, \dots$

To a triangle mesh, the basic test is <u>edge-triangle intersection</u> test: easy and robust

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228173830786.png" alt="image-20211228173830786" style="zoom:67%;" />

(The $t$ means the <u>interpolation value</u> between the points other than time)

##### Tunneling

The difference between collision and intersection detection:

- Intersection detection only considers if the 2 triangles (/…) have intersection at the two discrete time steps
- Collision detection can consider the penentration

<img src="C:/Users/TR/AppData/Roaming/Typora/typora-user-images/image-20211228174514285.png" alt="image-20211228174514285" style="zoom:50%;" />

**Problem**: The green triangle at state $\vb{x}^{[0]}$ and $\vb{x}^{[1]}$ doesn’t have intersection with the blue one but actually the position relationship changes => tunneling problem (penetrating through each other)

Not usually occurs for bulky objects but can happen for thin surfaces

#### Continuous Collision Detection (CCD)

CCD tests if any intersection exists <u>between two states</u>: $\vb{x}^{[0]}$ and $\vb{x}^{[1]}$ 

To a triangle mesh, there two basic tests: <u>vertex-triangle</u> and <u>edge-edge</u> tests 

**Vertex-Triangle**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228174836931.png" alt="image-20211228174836931" style="zoom:67%;" />

($\vb{x}_{30} = \vb{x}_3 - \vb{x}_{0}$ which is a function about time) 

Need to solve a third order equation (with one variable) => has formulae solution (not recommended, need to solve cubic root) / use binary division (0, 1) and get the first sol

**Edge-Edge**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211228212022858.png" alt="image-20211228212022858" style="zoom:67%;" />

**Issues** with CCD:

- <u>Floating-point errors</u>, especially due to root finding of a cubic equation
  - Buffering epsilons, but that causes <u>false positives</u>
  - Gaming GPUs often use <u>single floating-point precision</u>
- <u>Computational costs</u>: more expensive than DCD
  - Some argue that broad-phase collision culling is the bottleneck
- Difficulty in implementation

## Continuous Collision Response Approaches

### The Two Continuous Collision Response Approaches

Given the calculated next state $\vb{x}^{[1]}$, we want to update it into $\overline{\vb{x}}^{[1]}$, the path from $\vb{x}^{[0]}$ to $\overline {\vb{x}}^{[1]}$ is intersection-free

- **Interior Point Method**: Use the rest state to the collided state to estimate the every next state (safer)
  - Slow: far from solution; all vertices; cautiously by small step size
  - Always succeed
- **Impact Zone Method**: Use the final point to estimate, then do optimization in the “impact zone” 
  - Fast: Close to solution; only vertices in collision (impact zones); can take large step sizes
  - May not succeed

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229005436625.png" alt="image-20211229005436625" style="zoom: 67%;" />

### Interior Point Methods (Cont.)

#### Log-Barrier Interior Point Methods

For simplicity, just consider the Log-barrier repulsion between 2 vertices (usually in graphics do a cutoff at some distance)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229010339561.png" alt="image-20211229010338910" style="zoom:67%;" />
$$
E(\vb{x}) = -\rho \log\|\vb{x}_{ij}\|\\
\vb{f}_i(\vb{x}) = -\grad_iE = \rho \frac{\vb{x}_{ij}}{\|\vb{x}_{ij}\|^2}\ ;
\quad
\vb{f}_j(\vb{x}) = -\grad_jE = - \rho \frac{\vb{x}_{ij}}{\|\vb{x}_{ij}\|^2}
$$

#### Implementation

Formulate the problem:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229010911256.png" alt="image-20211229010911256" style="zoom: 67%;" />
$$
\overline{\vb{x}}^{[1]}\leftarrow \operatorname{argmin} _\vb{x}\left(\frac{1}{2}\|\vb{x-x}^{[1]}\|^2 - \rho\sum\log\|\vb{x}_{ij}\| \right)
$$
**Gradient Descent**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229011306261.png" alt="image-20211229011306261" style="zoom:80%;" />

The step size $\alpha$ must be adjusted to ensure that no collision happens on the way. To find $\alpha$, we need collision tests.

### Impact Zone Optimization (Cont.)

The goal of impact zone optimization is to optimize $\vb{x}^{[1]}$ until it becomes intersection-free. (This potentially suffers from the tunneling issue, uncommon) 

$$
\overline{\vb{x}}^{[1]} \leftarrow \operatorname{argmin}_\vb{x} \frac{1}{2}\|\overline{\vb{x}}-{\vb{x}}^{[1]}\|^2
$$
Such that:
$$
 \begin{cases}C(\mathbf{x})=-\left(\mathbf{x}_{3}-b_{0} \mathbf{x}_{0}-b_{1} \mathbf{x}_{1}-b_{2} \mathbf{x}_{1}\right) \cdot \mathbf{N} \leq 0 & \text { For each detected vertex-triangle pair } \\ C(\mathbf{x})=-\left(b_{2} \mathbf{x}_{2}+b_{3} \mathbf{x}_{3}-b_{0} \mathbf{x}_{0}-b_{1} \mathbf{x}_{1}\right) \cdot \mathbf{N} \leq 0 & \text { For each detected edge-edge pair }\end{cases}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229012241218.png" alt="image-20211229012241218" style="zoom:67%;" />

The optimizations are always trying to get close to the solution

### Rigid Impact Zones

The rigid impact zone method simply <u>freezes vertices</u> in collision from moving in their pre-collision state. It’s <u>simple and safe</u>, but has <u>noticeable artifacts</u>

Only used when all other methods not work

### A Practical System

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229012717208.png" alt="image-20211229012717208" style="zoom: 67%;" />

## Untangling Cloth (Discrete)

### Intersection Elimination

> Consider how to <u>eliminate existing intersections</u>, but without using any collision history

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229013144634.png" alt="image-20211229013144634" style="zoom:67%;" />

Useful when there are already intersections in simulation, due to:

- Past collision handling failures
- Intense user interaction

In this case, we don’t require the simulation is to always <u>intersection-free</u>

Eliminating <u>cloth-volume</u> and <u>volume-volume</u> intersections is straightforward: simply **pushing vertices/edges in the volume out**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229212934397.png" alt="image-20211229212934397" style="zoom: 67%;" />

### Untangling Cloth

<u>Cloth-cloth intersection</u> is complicated (don’t have a clear definition of inside and outside)

- Baraff et al. used flood-fill to segment cloth into <u>regions</u> and decided <u>which region is in intersection</u>. (Cannot handle boundary well, not friendly in GPU)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229214002651.png" alt="image-20211229214002651" style="zoom:50%;" />

  divide into several regions => which is larger => the smaller part moves to the larger side

- Volino and Magnenat-Thalmann proposed to untangle cloth by reducing the intersection contour

  Their method can handle boundaries, but it doesn’t always work

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211229214433087.png" alt="image-20211229214433087" style="zoom: 67%;" />



# Lecture 10 Surface Waves

**Two Types of Simulation Approaches**

- **Lagrangian** Approach: Dynamic particles or mesh (Node movement carries physical quantities (mass, velocity, …))
- **Eulerian** Approach: Static grid or mesh (Grid/Mesh doesn’t move; Stored physical quantities change)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/V4rJQGzKxXwWhCM.png" alt="image-20220104125629085" style="zoom: 67%;" />

## A Height Field Model

### Height Field

In 2D, a (1.5D) height field is a height function $h(x)$ (cannot have 2 $h$ at one position) and the velocity is also a function $u(x)$  (negative -> opposite direction)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104165850951.png" alt="image-20220104165850951" style="zoom: 50%;" />

The height field and velocity can be represented by:  ($h\cdot u$ means the volume flowing through a position at the unit time)
$$
\left\{\begin{aligned}
&\frac{\mathrm{d}h(x)}{\mathrm{d}t} + \frac{\mathrm{d}(h(x)u(x))}{\mathrm{d}x} = 0\\
&\frac{\mathrm{d}u(x)}{\mathrm{d}t} = \underbrace{- u(x) \frac{\mathrm{d}u(x)}{\mathrm{d}x}}_{\text{advection}} - \frac{1}{\rho}\frac{\mathrm{d}P(x)}{\mathrm{d}x} + \underbrace{a(x)}_{\text{ext.}}
\end{aligned}
\right.
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104170150684.png" alt="image-20220104170150684" style="zoom:50%;" />

Ignoring <u>advection</u> and <u>external acceleration</u>, we get a simple form of velocity derivative: ($\rho$ - density; $P(x)$ - pressure)
$$
\frac{\mathrm{d}u(x)}{\mathrm{d}t} = -\frac{1}{\rho}\frac{\mathrm{d}P(x)}{\mathrm{d}x} 
$$
From $x$ to $x+\mathrm{d}x$: $\mathrm{d}P(x) = P(x+\mathrm{d}x) - P(x)$ => the velocity changes due to the change of pressure (also dep on density)

$u(x)$ increases when $\mathrm{d}P(x)<0$ and increases when $\mathrm{d}P(x)>0$ 

### Shallow Wave Equation

Simplification of the two equations by assuming <u>shallow</u> wave (the wave is small and the <u>height change can be neglected</u>)  
$$
\left\{\begin{aligned} 
&\frac{\mathrm{d}h}{\mathrm{d}t} + \frac{\mathrm{d}(hu)}{\mathrm{d}x} = 0 \longrightarrow \frac{\mathrm{d}h}{\mathrm{d}t} + \cancel{ u \frac{\mathrm{d}h}{\mathrm{d}x} } + h\frac{\mathrm{d}u}{\mathrm{d}x} = 0 \\
&\frac{\mathrm{d}u}{\mathrm{d}t} = - \frac{1}{\rho}\frac{\mathrm{d}P}{\mathrm{d}x} 
\end{aligned}
\right.
\begin{aligned} 
&\stackrel{\mathrm{d}t}\longrightarrow \\
&\stackrel{\mathrm{d}x}\longrightarrow 
\end{aligned} 
\left\{\begin{aligned} 
&  \frac{\mathrm{d}^2 h}{\mathrm{d}t^2} + h \frac{\mathrm{d}^2}{\mathrm{d}x\mathrm{d}t} = 0\\
&  \frac{\mathrm{d}^{2} u}{\mathrm{d} x \mathrm{d} t}=-\frac{1}{\rho} \frac{\mathrm{d}^{2} P}{\mathrm{d} x^{2}}
\end{aligned}
\right.
$$
Eliminate $u$ and formulate the **shallow wave equation** (indep on the velocity field)
$$
\Rightarrow \frac{\mathrm{d}^2h}{\mathrm{d}t^2}  = \frac{h}{\rho} \frac{\mathrm{d}^2P}{\mathrm{d}x^2} 
$$

### Finite Differencing

#### Discretization

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104173532131.png" alt="image-20220104173532131" style="zoom: 50%;" />

#### Finite Difference

Use the difference to approximate the derivative

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104173745771.png" alt="image-20220104173745771" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104173759400.png" alt="image-20220104173759400" style="zoom:67%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220104174219237.png" alt="image-20220104174219237" style="zoom: 67%;" />

- Forward differencing (first order, 2nd order error)
  $$
  f\left(t_{0}+\Delta t\right)=f\left(t_{0}\right)+\Delta t \frac{\mathrm{d} f\left(t_{0}\right)}{\mathrm{d} t}+\frac{\Delta t^{2} \mathrm{d}^{2} f\left(t_{0}\right)}{2}+\cdots \Rightarrow
  \frac{\mathrm{d}f(t_0)}{\mathrm{d}t} \approx \frac{f(t_0 + \Delta t)-f(t_0)}{\Delta t}
  $$

- Backward differencing (first order)
  $$
  f\left(t_{0}-\Delta t\right)=f\left(t_{0}\right)-\Delta t \frac{\mathrm{d} f\left(t_{0}\right)}{\mathrm{d} t}+\frac{\Delta t^{2} \mathrm{d}^{2} f\left(t_{0}\right)}{2}+\cdots \Rightarrow
  \frac{\mathrm{d}f(t_0)}{\mathrm{d}t} \approx \frac{f(t_0 )-f(t_0- \Delta t)}{\Delta t}
  $$

- **Central differencing** (second order)
  $$
  \Rightarrow \frac{\mathrm{d}f(t_0)}{\mathrm{d}t} \approx \frac{f(t_0+\Delta t)-f(t_0-\Delta t)}{2\Delta t}
  $$

**Second-Order Derivatives** (central differencing in the fluid field)

Applying central differencing twice to estimate <u>height</u> $\mathrm{d}^2h_i / \mathrm{d}t^2$ (Laplacian)
$$
\frac{\mathrm{d}^{2} h_{i}\left(t_{0}\right)}{\mathrm{d} t^{2}} \approx \frac{\frac{\mathrm{d} h_{i}\left(t_{0}+0.5 \Delta t\right)}{\mathrm{d} t}-\frac{\mathrm{d} h_{i}\left(t_{0}-0.5 \Delta t\right)}{\mathrm{d}t}}{\Delta t} \approx \frac{h_{i}\left(t_{0}+\Delta t\right)+h_{i}\left(t_{0}-\Delta t\right)-2 h_{i}\left(t_{0}\right)}{\Delta t^{2}}
$$
Also for <u>pressure</u> $\mathrm{d}^2P / \mathrm{d}x^2$
$$
\frac{\mathrm{d}^{2} P_{i}}{\mathrm{d} x^{2}} \approx \frac{\frac{\mathrm{d} P_{i+0.5}}{\mathrm{d} x}-\frac{\mathrm{d} P_{i-0.5}}{\mathrm{d} x}}{\Delta x} \approx \frac{P_{i+1}+P_{i-1}-2 P_{i}}{\Delta x^{2}}
$$

#### Discretized Shallow Wave Equation

Want to obtain the height field in the next time step
$$
\frac{\mathrm{d}^2h}{\mathrm{d}t^2} = \frac{h}{\rho} \frac{\mathrm{d}^2 P}{\mathrm{d} x^2} 
\Rightarrow
h_{i}\left(t_{0}+\Delta t\right)=2 h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)+\frac{\Delta t^{2} h_{i}}{\Delta x^{2} \rho}\left(P_{i+1}+P_{i-1}-2 P_{i}\right)
$$
**Problem**: Volume loss or generation

#### Volume Preservation

We want the volume to stay the same

Suppose $\sum h_i(t) = \sum h_i(t-\Delta t) = V$, but by using the formuation in the prev section, we will get
$$
\begin{aligned}
\sum h_i(t+\Delta t) &= 2 \cancel{\sum h_i(t_0)}^V - \cancel{\sum h_i(t_0-t) }^V + \sum \frac{\Delta t^2 h_i}{\Delta x^2 \rho} (P_{i+1} + P_{i-1} -2P_i) \\
&= V +  \sum \frac{\Delta t^2 h_i}{\Delta x^2 \rho} (P_{i+1} + P_{i-1} -2P_i)
\end{aligned}
$$
and we cannot guarantee $ \sum \frac{\Delta t^2 h_i}{\Delta x^2 \rho} (P_{i+1} + P_{i-1} -2P_i)$ is 0

##### Solution 1

Modify scheme into: (not using $h_i$ as the coefficient => using $\frac{h_{i-1}+h_i}{2}$ and $\frac{h_{i+1}+h_i}{2} $ instead) 
$$
\begin{aligned}h_{i}\left(t_{0}+\Delta t\right)&=2 h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)+\frac{\Delta t^{2} h_{i}}{\Delta x^{2} \rho}\left(P_{i+1}+P_{i-1}-2 P_{i}\right)\\
\Rightarrow 
h_{i}\left(t_{0}+\Delta t\right) &=2 h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)+\frac{\Delta t^{2}}{\Delta x^{2} \rho}\left(\left(\frac{h_{i-1}+h_{i}}{2}\right)\left(P_{i-1}-P_{i}\right)+\left(\frac{h_{i+1}+h_{i}}{2}\right)\left(P_{i+1}-P_{i}\right)\right)
\end{aligned}
$$
Can be understanded as the water exchanges between $h_i$ and $h_{i+1}$. The water flowing from the LHS will flow into the RHS => Volume preserved

##### Solution 2

Assuming $$h_i$$ in the right term is const
$$
\begin{aligned}
\Rightarrow h_{i}\left(t_{0}+\Delta t\right) &=2 h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)+\frac{\Delta t^{2} H}{\Delta x^{2} \rho}\left(P_{i+1}+P_{i-1}-2 P_{i}\right)\\
\text{So that } \sum h_{i}\left(t_{0}+\Delta t\right)&= V +\frac{\Delta t^{2} H}{\Delta x^{2} \rho}\sum\underbrace{\left((P_{i+1} -P_{i})+(P_{i-1}-P_{i})\right)}_{\text{Must be 0}}
\end{aligned}
$$

### Pressure

The pressure is related to the water **height**: $P_i = \rho g h_i$
$$
\Rightarrow h_{i}\left(t_{0}+\Delta t\right)=2 h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)+\underbrace{\frac{\Delta t^{2} H g}{\Delta x^{2}}}_{\alpha}\left(h_{i+1}\left(t_{0}\right)+h_{i-1}\left(t_{0}\right)-2 h_{i}\left(t_{0}\right)\right)
$$
Replace the term with a constant $\alpha$ 

### Viscosity

Viscosity tends to slow down the waves (Add a viscosity constant $\beta$: too high: not viscos; too low: not like fluid)
$$
h_{i}\left(t_{0}+\Delta t\right)=h_{i}\left(t_{0}\right)+\beta \underbrace{\left(h_{i}\left(t_{0}\right)-h_{i}\left(t_{0}-\Delta t\right)\right)}_{\text{Momentum here}}+\alpha\left(h_{i+1}\left(t_{0}\right)+h_{i-1}\left(t_{0}\right)-2 h_{i}\left(t_{0}\right)\right)\\
$$

### Boundary Conditions

- **Dirichlet** Boundary: The boundary height $H_{i+1}$ is constant (open boundary)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105171719986.png" alt="image-20220105171719986" style="zoom:67%;" />
  $$
  h_{i+1} - h_i + h_{i-1} - h_i = H_{i+1} - h_i + h_{i-1} - h_i
  $$

- **Neumann** Boundary: specifies the boundary <u>derivatives</u>. For example, a <u>zero-derivative</u> boundary (no water exchange) means $h_{i+1}\equiv h_i$ (closed boundary)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105172434946.png" alt="image-20220105172434946" style="zoom:67%;" />
  $$
  h_{i+1} - h_i + h_{i-1} -h_i = h_{i-1} - h_i
  $$
  

#### Algorithm with Neumann Boundary

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105172628214.png" alt="image-20220105172628214" style="zoom:67%;" /> – to 3D – <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105172707786.png" alt="image-20220105172707786" style="zoom:67%;" />

## Two-way Coupling

The coupling will be two ways: between liquid->solid and solid->liquid (also can be bubbles / soft solids / …)

**Key**: how to expel water out of the gray cell regions

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105173316343.png" alt="image-20220105173316343" style="zoom:40%;" />  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105173333356.png" alt="image-20220105173333356" style="zoom: 40%;" />

### Virtual Height

**Idea**: Set up a virtual height $v_i$ so that $h_i^{\text{real\_new}} = h_i - e_i$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105173538947.png" alt="image-20220105173538947" style="zoom:67%;" />
$$
\left\{\begin{aligned}
\text{Left: }  \ & h_{i}-e_{i}=h_{i}+\beta\left(h_{i}-h_{i}^{\text {old }}\right)+\alpha\left(\nu_{i+1}+h_{i+1}+h_{i-1}-2\nu_{i}-2 h_{i}\right)=h_{i}^{\text {new }}+\alpha\left(\nu_{i+1}-2 \nu_{i}\right)\\
\text{Right: } \ & h_{i+1}-e_{i+1}=h_{i+1}+\beta\left(h_{i+1}-h_{i+1}^{\text {old }}\right)+\alpha\left(h_{i+2}+\nu_{i}+h_{i}-2 \nu_{i+1}-2 h_{i+1}\right)=h_{i+1}^{\text {new }}+\alpha\left(\nu_{i}-2 \nu_{i+1}\right)
\end{aligned}\right.
$$

### Poisson’s Equation

The outcome is Poisson’s equation, with $\nu_i$ and $\nu_{i+1}$ being unknowns 
$$
\begin{aligned}
2 \nu_{i}-\nu_{i+1} &=\frac{1}{\alpha}\left(h_{i}^{\text{new}}-h_{i}+e_{i}\right)=b_{i} \\
-\nu_{i}+2 \nu_{i+1} &=\frac{1}{\alpha}\left(h_{i+1}^{\text{new}}-h_{i+1}+e_{i+1}\right)=b_{i+1}
\end{aligned}
$$
Regard as a Laplacian operator for every equation ($\nu_{i-1} $ & $\nu_{i+2}$ are all 0, no virtual height)
$$
\begin{bmatrix} 2 &-1 \\-1 & 2 \end{bmatrix} 
\begin{bmatrix}\nu_i \\\nu_{i+1} \end{bmatrix} = 
\begin{bmatrix} b_i \\ b_{i+1} \end{bmatrix}
\Rightarrow
\left[\begin{array}{cccc}
1 & & & \\
-1 & 2 & -1 &  \\
 & -1 & 2 & -1 \\
 & & 1 & 1
\end{array}\right]=\left[\begin{array}{c}
\nu_{i-1} \\
\nu_{i} \\
\nu_{i+1} \\
\nu_{i+2}
\end{array}\right]=\left[\begin{array}{c}
0 \\
b_{i} \\
b_{i+1} \\
0
\end{array}\right]
$$

### Algorithm

**Process**: Set <u>boundary condition</u> -> <u>Mask</u> (which grids need solving) -> `PCG_Solve()` to retrun $\nu$ -> Update the <u>height field</u> (Usually need to multiply a factor, prevent too large waves when dragging the cube <- explicit’s unstability) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105174829562.png" alt="image-20220105174829562" style="zoom:67%;" />

### Rigid Body Update

Estimate the floating (buoyancy) force by the actual water expelled in every column

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20220105211306589.png" alt="image-20220105211306589" style="zoom:67%;" />
$$
f_i = \rho g \Delta x(h_i - h_i^{\text{new}})\quad& \text{in 2D}\\
f_{i,j} = \rho g \Delta A (h_{i,j} - h_{i,j}^{\text{new}})\quad& \text{in 3D}
$$
Also need to consider rotation => torque
