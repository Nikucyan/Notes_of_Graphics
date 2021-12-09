# Taichi - Physics Engines

[![](https://img.shields.io/badge/Home%20Page-%20%20-blueviolet)](https://nikucyan.github.io/) [![](https://img.shields.io/badge/Repo-%20%20-blue)](https://github.com/Nikucyan/Notes_of_Graphics/tree/main/Taichi) 

- **Games 201 - ADVANCED PHYSICS ENGINES**

  *Lecturer: Dr. Yuanming Hu*

- **Taichi Graphics Course S1** (for completion and new feature updates)

  *Lecturer: Dr. Tiantian Liu*



> **Some supplement contents about Taichi**
>
> If overlapping contents: just add on the original GAMES201 notes. The new contents: add at the last of the notes individually.
>
> - Exporting Results (Lecture 2)
> - OOP and Metaprogramming (Lecture 3)
> - Diff. Programming (Lecture 3)
> - Visualization (Lecture 3)





# Lecture 1  Introduction



Keyword: Discretization / Efficient solvers / Productivity / Performance / Hardware architecture / Data structures / Parallelization / Differentiability (DiffTaichi)



## Taichi Programming Language

### Initialization

Required every time using taichi

Use `ti.init`, for spec. cpu or gpu: use `ti.cpu` (default) or `ti.gpu`

```python
ti.init(arch=ti.cuda) # spec run in any hardware
```

### Data

#### Data Types

signed integers `ti.i8` (~ 64, default `i32`) / unsigned integers `ti.u8` (~64) / float-point numbers `ti.f32`(default) (~ 64)

##### Modify Default

The default can be changed via `ti.init`

- `ti.init(default_fp=ti.f32)`
- `ti.init(default_ip=ti.i64)` 

##### Type Promotions

- i32 + f32 = f32
- i32 + i64 = i64

Switch to high percision automatically

##### Type Casts

- Implicit casts: Static types within the Taichi Scope

  ``` python
  def foo():	# Directly inside Python scope
      a = 1
      a = 2.7	# Python can re-def types automatically
      print(a)	# 2.7
  ```

  ``` python
  @ti.kernel	# Inside Taichi scope
  def foo():
      a = 1	# already def as a int type
      a = 2.7	# re-def failed (in Taichi)
      print(a)	# 2
  ```

- Explicit casts: `variable = ti.casts(variable, type)`

  ``` python
  @ti.kernel
  def foo():
      a = 1.8
      b = ti.cast(a, ti.i32)	# switch to int;	b = 1
      c = ti.cast(b, ti.f32)	# switch to floating;	c = 1.0
  ```

#### Tensor

Multi-dimensional arrays（高维数组）

- **Self-defines**: Use `ti.types` to create compound types including vector / matrix / struct

  ``` python
  import taichi as ti
  ti.init(arch = ti.cpu)
  
  # Define your own types of data
  vec3f = ti.types.vector(3, ti.f32)	# 3-dim
  mat2f = ti.types.matrix(2, 2, ti.f32)	# 2x2
  ray = ti.types.struct(ro = vec3f, rd = vec3f, l = ti.f32)
  
  @ti.kernel
  def foo():
      a = vec3f(0.0)
      print(a)	# [0.0, 0.0, 0.0]
      d = vec3f(0.0, 1.0, 0.0)
      print(d)	# [0.0, 1.0, 0.0]
      B = mat2f([[1.5, 1.4], [1.3, 1.2]])	
      print("B = ", B)	# B = [[1.5, 1.4], [1.3, 1.2]]
      r = ray(ro = a, rd = d, l = 1)
      print("r.ro = ", r.ro)	# r.ro = [0.0, 0.0, 0.0]
      print("r.rd = ", r.rd)	# r.rd = [0.0, 1.0, 0.0]
      
  foo()
  ```

- **Pre-defines**: An element of the tensors can be either a scalar (`var`), a vector (`ti.Vector`) or a matrix (`ti.Matrix`) (`ti.Struct`)

  Accessed via `a[i, j, k]`syntax (no pointers)

  ```python
  import taichi as ti
  ti.init()
  a = ti.var(dt=ti.f32, shape=(42, 63)) # A tensor of 42x63 scalars
  b = ti.Vector(3, dt=ti.f32, shape=4) # A tensor of 4x 3D vectors (3 - elements in the vector, shape - shape of the tensor, composed by 4 3D vectors)
  C = ti.Matrix(2, 2, dt=ti.f32, shape=(3, 5)) # A tensor of 3x5 2x2 matrices
  loss = ti.var(dt=ti.f32, shape=()) # A 0-D tensor of a single scalar (1 element)
  
  a[3, 4] = 1
  print('a[3, 4] = ', a[3, 4]) # a[3, 4] = 1.000000
  b[2] = [6, 7, 8] 
  print('b[0] =', b[0][0], b[0][1], b[0][2]) # b[0] not yet supported (working)
  loss[None] = 3
  print(loss[None]) # 3
  ```

#### Field

`ti.field`: A global N-d array of elements

​	`heat_field = ti.field(dtype=ti.f32, shape=(256, 256))`

- **global**: a field can be read / written from both Taichi scope and Python scope

- **N-d**: Scalar (N = 0); Vector (N = 1); Matrix (N = 2); Tensor (N = 3, 4, 5, …)

- **elements**: scalar, vector, matrix, struct

- access elements in a field using [i, j, k, …] indexing

  ``` python
  import taichi as ti
  ti.init(arch=ti.cpu)
  
  pixels = ti.field(dtype=float, shape=(16, 8))
  pixels[1,2] = 42.0	# index the (1,2) pixel on the screen
  ```

  ``` python
  import taichi as ti
  ti.init(arch=ti.cpu)
  
  vf = ti.Vector.field(3, ti.f32, shape=4)	# 4x1 vectors, every vector is 3x1
  
  @ti.kernel
  def foo():
      v = ti.Vector([1, 2, 3])
      vf[0] = v
  ```

- **Special Case**: access a 0-D field using `[None]`

  ``` python
  zero_d_scalar = ti.field(ti.f32, shape=())
  zero_d_scalar[None] = 1.5	# Scalar in the scalar field
  
  zero_d_vector = ti.Vector.field(2, ti.f32, shape=())
  zero_d_vector[None] = ti.Vector([2.5, 2.6])
  ```

**Other Examples**:

- 3D gravitational field in a 256x256x128 room

  `gravitational_field = ti.Vector.field(n=3, dtype=ti.f32, shape=(256, 256, 128))`

- 2D strain-tensor field in a 64x64 grid

  `strain_tensor_field = ti.Matrix.field(n = 2, m = 2, dtype = ti.f32, shape = (64, 64))`

- a global scalar that want to access in a taichi kernel

  `global_scalar = ti.field(dtype = ti.f32, shape=())`

### Kernels

Must be decorated with `@ti.kernel` (Compiled, statically-typed, lexically-scoped, parallel and differentiable - faster)

For loops at the **outermost scope** in a Taichi kernel is **automatically parallelized** (if inside - serial)

​	If a outermost scope is not wanted to parallelize for and an inside scope is, write the unwanted one in the python scope and call the other one as the outermost in the kernel.

#### Arguments

- At most 8 parameters

- Pass from Python scope to the Taichi scope

- Must be type-hinted

  ``` python
  @ti.kernel
  def my_kernel(x: ti.i32, y: ti.f32):	# explicit input variables with types
      print(x + y)
      
  my_kernel(2, 3.3)	# 5.3    
  ```

- Scalar only (if vector needs to input separately)

- Pass by value

  Actually copied from the var. in the Python scope and if the values of some var. `x` is modified in the Taichi kernel, it won’t change in the Python scope.

#### Return Value

- May or may not return

- Returns one single scalar value only

- Must be type-hinted

  ``` python
  @ti.kernel
  def my_kernel() -> ti.i32:	# returns i32
      return 233.666
  
  print(my_kernel())	# 233	(casted to int32)
  ```

### Functions

Decorated with `@ti.func`, usually for high freq. used func.

Taichi's function can be called in taichi's kernel but can't be called by python (not global)

Function 可以被 kernel 调用但不能被 py 调用，kernel 也不能调用 kernel，但是 function可以调用 function

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211005212840697.png" alt="image-20211005212840697" style="zoom: 50%;" />

- Taichi functions can be nested (function in function)
- Taichi functions are force-inlined (cannot iterate)

#### Arguments

- Don’t need to be type-hinted

- Pass by value (for the force-inlined `@ti.func`)

  if want to pass outside, use `return`

### Scalar Math

Similar to python. But Taichi also support **chaining comparisons** (`a < b <= c != d`)

Specifically, `a / b` outputs floating point numbers, `a // b` ouputs integer number (as in py3)

### Matrices and Linear Algebra

`ti.Matrix` is for small matrices only. For large matrices, 2D tensor of scalars should be considered.

Note that element-wise product in taichi is `*` and mathematically defined matrix product is `@` (like MATLAB)

Common Operations:（返回矩阵 A 经过变换的结果而非变换 A 本身）

```python
A.transpose() # 转置
A.inverse() # 逆矩阵
A.trace() # 迹
A.determinant(type)
A.normalized() # 向量除以自己模长
A.norm() # 返回向量长度
A.cast(type) # 转换为其他数据类型
ti.sin(A)/cos(A) # element wise（给 A 中所有元素都做 sin 或者 cos 运算）
```

### Parallel for-loops

(Automatically parallalized)

#### Range-for loops 

Same as python for loops

#### Struct-for loops

Iterates over (sparse) tensor elements. Only lives at the outermost scope looping over a `ti.field`

​	`for i,j in x:` can be used

Example:

```python
import taichi as ti

ti.init(arch=ti.gpu)  # initiallize every time

n = 320
pixels = ti.var(dt=ti.f32, shape=(n*2, n))  # every element in this tensor is 32 bit float point number，shape 640x320

@ti.kernel
def paint(t: ti.f32):
	for i, j in pixels:  # parallized over all pixel
        pixels[i, j] = i * 0.001 + j * 0.002 + t 
        # This struct-for loops iterate over all tensor coordinates. i.e. (0,0), (0,1), (0,2), ..., (0,319), (1,0), ..., (639,319)
        # Loop for all elements in the tensor
        # For this dense tensor, every element is active. But for sparse tensor, struct-for loop will only work for active elements
paint(0.3)
```

- `break` is NOT supported in the parallel for-loops

### Atomic Operations

In Taichi, augmented assignments (e.g. x[i] += 1) are automatically atomic.

(+=: <u>the value on the RHS is directly summed on the current value of the var. and the referncing var.(array)</u>)

(Atomic operation: <u>an operation that will always be executed without any other process being able to read or change state that is read or changed during the operation. It is effectively executed as a single step, and is an important quality in a number of algorithms that deal with multiple independent processes, both in synchronization and algorithms that update shared data without requiring synchronization.</u>)

When modifying global var. in parallel, make sure to use atomic operations.

```python
@ti.kernel
def sum():  # to sum up all elements in x
    for i in x:
        # Approach 1: OK
        total[None] += x[i]	 # total: 0-D tensor => total[None]
        
        # Approach 2: OK
        ti.atomic_add(total[None], x[i])  # will return value before the atomic addition
        
        # Approach 3: Wrong Result (Not atomic)
        total[None] = total[None] + x[i]	# other thread may have summed
```

### Taichi-scope vs. Python-scope

**Taichi-scope**: Everything decorated with `ti.kernel` and `ti.func`

​	Code in Taichi-scope will be compiled by the Taichi <u>compiler</u> and run on parallel devices (Attention that in parallel devices the order of `print` may not be guaranteed)

- Static data type in the Taichi scope

  The type won’t change even if actually defines some other values / fields (error)

- Static lexical scope in the Taichi scope

  ``` python
  @ti.kernel
  def err_out_of_scope(x:float):
      if x < 0:	# abs value of x
          y = -x
      else:
          y = x
      print(y)  	# y generated in the 'if' and not pass outside, error occurs
  ```

- Compiled JIT (just in time) (cannot see Python scope var. at run time)

  ``` python
  a = 42
  
  @ti.kernel
  def print_a():
      print('a = ', a)
  
  print_a()	# 'a = 42'
  a = 53
  print('a = ', a)	# 'a = 53'
  print_a()	# still 'a = 42'
  ```

  Another demo:

  ``` python
  d = 1
  
  @ti.kernel
  def foo():
      print('d in Taichi scope = ', d)
      
  d += 1	# d = 2
  foo()	# d in Taichi scope = 2 (but after this call, ti kernel will regard d as a constant)
  d += 1 	# d = 3
  foo()	# d in Taichi scope = 2 (d not changed in Ti-scope but changed in Py-scope)
  ```
  
  If want real global: use `ti.field`
  
  ``` python
  a = ti.field(ti.i32, shape=())
  
  @ti.kernel
  def print_a():
  	print('a=', a[None])
  
  a[None] = 42
  print_a() # "a= 42"
  a[None] = 53
  print_a() # "a= 53"
  ```

**Python-scope**: Code outside the Taichi-scope

​	Code in Python-scope is simple py code and will be executed by the py interpreter

### Phases

1. Initialization: `ti.init()`

2. Tensor allocation: `ti.var`, `ti.Vector`, `ti.Matrix` 

   ​	<u>Only define tensors in this allocation phase and never define in the computation phase</u>

3. Computation (launch kernel, access tensors in Py-scope)

4. Optional: Restart (clear memory, destory var. and kernels): `ti.reset()`

### Practical Example  (fractal)

```python
import taichi as ti

ti.init(arch=ti.gpu)

n = 320
pixels = ti.var(dt=ti.f32, shape=(n * 2, n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])	# calculate the square of a complex number
	# In this example, use a list and put in [] to give values for ti.Vector 

@ti.kernel
def paint(t: ti.f32):	# time t - float point 32 bit
    for i, j in pixels:  # parallized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2	 # Julia set formula
        iterations = 0	 # iterate for all pixels
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c	# user-defined in @ti.func
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02
        
gui = ti.GUI("Julia Set", res = (n*2, n))	# ti's gui

for i in range(1000000):
    paint(i * 0.03)
    gui.set_image(pixels)
    gui.show
```

### Debug Mode



# Lecture 2  Lagrangian  View



## Two Views (Def)

### Lagrangian View

Sensors that move passively with the simulated material（节点跟随介质移动）

粒子会被额外记录位置，会随（被模拟的）介质不断移动

### Euler View

Still sensors that never moves（穿过的介质的速度）

网格中每个点的位置不会被特别记录

## Mass-Spring Systems

弹簧 - 质点模型 (Cloth / Elastic objects / ...)

### Hooke's Law

$$
\begin{aligned}
\mathbf{f}_{i j}& =-k\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}-l_{i j}\right)\left(\widehat{\mathbf{x}_{i}-\mathbf{x}}_{j}\right)\\
\mathbf{f}_i & = \sum^{j\neq i}_{j} \mathbf{f}_{ij}
\end{aligned}
$$

(f - force, k - Spring stifness, L~i,j~ - Spring rest length between particle i and j, ^ - normalization, $\widehat{\mathbf{x}_{i}-\mathbf{x}}_{j}$​ - direction vector of paticle i to j)

### Newton's Second Law of Motion

$$
\frac{\partial \mathbf{v}_i}{\partial t} = \frac{1}{m_i} \mathbf{f}_i \, ;
\quad
\frac{\partial \mathbf{x}_i}{\partial t} = \mathbf{v}_i
$$

## Time Integration

### Common Types of Integrators

- Forward Euler (explict)

  $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \frac{\mathbf{f}_t}{m}$,  $\mathbf{x}_{t+1}=\mathbf{x}_{t}+\Delta t \mathbf{v}_{t}$

- Semi-implicit Euler (aka. Symplectic Euler, Explicit)

  $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \frac{\mathbf{f}_t}{m}$,  $\mathbf{x}_{t+1}=\mathbf{x}_{t}+\Delta t \mathbf{v}_{t+1}$

  （准确性上提升，以 $t+1$​​​ 步的 $\mathbf{v}$​​​ 替代第 $t$​​ 步的使用）

- Backward Euler (often with Newton's Method, Implicit)

  $\mathbf{x}_{t+1}=\mathbf{x}_{t}+\Delta t \mathbf{v}_{t+1}$​,  $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \textbf{M}^{-1} \mathbf{f}(\mathbf{x}_{t+1})$​ （相互依赖）

  Full implement see later

### Comparison

- Explicit: (forward Euler / symplectic Euler / RK (2, 3, 4) / ...)

  Future depends only on past, easy to implement, easy to <u>explode</u>, bad for stiff materials

  $\Delta t \leq c\sqrt{\frac{m}{k}}\; (c\sim 1)$​（当粒子质量变大的时候允许的 $\Delta t$ 就会变大，$k$​（硬度）越大的时候允许的也会变小 - 不适合硬度大的）

  ~ 数值稳定性，不会衰减而是随时间指数增长

- Implicit (back Euler / middle-point)

  Future depends on both future and past, harder to implement, need to solve a syustem of (linear) equation, hard to optimize, time steps become more expensive but time steps are larger, extra numerical damping and locking (overdamping) (but generally, uncoonditionally stable)
  
  （显式 - 容易实现，数值上较为不稳定，受 dt 步长影响较大；隐式 - 难实现，允许较长步长）

### Implementing a Mass-Spring System with Symplectic Euler

#### Steps

- Compute new velocity using $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \frac{\mathbf{f}_t}{m}$
- Collision with ground
- Compute new position using $\mathbf{x}_{t+1}=\mathbf{x}_{t}+\Delta t \mathbf{v}_{t+1}$

#### Demo

```python
import taichi as ti

ti.init(debug=True)

max_num_particles = 256

dt = 1e-3

num_particles = ti.var(ti.i32, shape=())
spring_stiffness = ti.var(ti.f32, shape=())
paused = ti.var(ti.i32, shape=())
damping = ti.var(ti.f32, shape=())

particle_mass = 1
bottom_y = 0.05

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))

connection_radius = 0.15

gravity = [0, -9.8]

@ti.kernel
def substep():	# 每一个新的时间步，把每一帧分为若干步（在模拟中 time_step 需要取相应较小的数值）
    
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):	# 枚举全部 i
        v[i] *= ti.exp(-dt * damping[None]) # damping
        total_force = ti.Vector(gravity) * particle_mass  # 总受力 G = mg
        for j in range(n):	# 枚举其余所有粒子 是否有关系
            if rest_length[i, j] != 0:	# 两粒子之间有弹簧？
                x_ij = x[i] - x[j]
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()	
                # 胡克定律公式 -k * ((xi - xj).norm - rest_length) * (xi - xj).norm
        v[i] += dt * total_force / particle_mass	# sympletic euler: 用力更新一次速度
        
    # Collide with ground （计算完力和速度之后立刻与地面进行一次碰撞）
    for i in range(n):
        if x[i].y < bottom_y:	# 一旦发现有陷入地下的趋势就把速度的 y component 设置为 0
            x[i].y = bottom_y
            v[i].y = 0

    # Compute new position
    for i in range(num_particles[None]):
        x[i] += v[i] * dt	# 新的位置的更新：把速度产生的位移累加到位置上去

        
@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32): # Taichi doesn't support using Matrices as kernel arguments yet
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    
    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1
    
    
gui = ti.GUI('Mass Spring System', res=(512, 512), background_color=0xdddddd)

spring_stiffness[None] = 10000
damping[None] = 20

new_particle(0.3, 0.3)
new_particle(0.3, 0.4)
new_particle(0.4, 0.4)

while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1
                
    if not paused[None]:
        for step in range(10):
            substep()
    
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    
    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)
    
    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x445566)
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.show()
```

### Backward Euler Implicit Implement

#### Steps

- $\mathbf{x}_{t+1}=\mathbf{x}_{t}+\Delta t \mathbf{v}_{t+1}$​,  $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \textbf{M}^{-1} \mathbf{f}(\mathbf{x}_{t+1})$​ （相互依赖）
- **Eliminate $\mathbf{x}_{t+1}$**: $\mathbf{v}_{i+1} = \mathbf{v}_t + \Delta t \textbf{M}^{-1} \mathbf{f}(\mathbf{x}_{t}+\Delta t\mathbf{v}_{t+1})$​ （f 通常为非线性，用下述泰勒展开一次线性化）
- **Linearize (Newton's)**: $\mathbf{v}_{t+1}=\mathbf{v}_{t}+\Delta t \mathbf{M}^{-1}\left[\mathbf{f}\left(\mathbf{x}_{t}\right)+\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\left(\mathbf{x}_{t}\right) \Delta t \mathbf{v}_{t+1}\right]$​
- **Clean up**: $\left[\mathbf{I}-\Delta t^{2} \mathbf{M}^{-1} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\left(\mathbf{x}_{t}\right)\right] \mathbf{v}_{t+1}=\mathbf{v}_{t}+\Delta t \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}_{t}\right)$​ (Linear System)
- To solve the equation: Jacobi / Gauss-Seidel iteration OR conjugate gradients

**Solving the system**:
$$
\begin{aligned} \mathbf{A} &=\left[\mathbf{I}-\Delta t^{2} \mathbf{M}^{-1} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\left(\mathbf{x}_{t}\right)\right] \\ \mathbf{b} &=\mathbf{v}_{t}+\Delta t \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}_{t}\right) \\ \mathbf{A} \mathbf{v}_{t+1} &=\mathbf{b} \end{aligned}
$$

#### Demo

```python
import taichi as ti
import random

ti.init()

n = 20	

A = ti.var(dt=ti.f32, shape=(n, n))	 # 20 x 20 Matrix
x = ti.var(dt=ti.f32, shape=n)
new_x = ti.var(dt=ti.f32, shape=n)
b = ti.var(dt=ti.f32, shape=n)

@ti.kernel	# iteration kernel
def iterate():
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] * x[j]
                
        new_x[i] = r / A[i, i]	# 每次都更新 x 使得 x 能满足矩阵一行或一个的线性方程组（局部更新）- 对性质好的矩阵逐渐收敛
        
    for i in range(n):
        x[i] = new_x[i]

@ti.kernel	# Compute residual b - A * x
def residual() -> ti.f32:	# residual 一开始会非常大 经过若干次迭代会降到非常低
    res = 0.0
    
    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] * x[j]
        res += r * r
        
    return res

for i in range(n):	# 初始化矩阵
    for j in range(n):
        A[i, j] = random.random() - 0.5

    A[i, i] += n * 0.1
    
    b[i] = random.random() * 100

for i in range(100):	# 执行迭代
    iterate()	
    print(f'iter {i}, residual={residual():0.10f}')
    

for i in range(n):
    lhs = 0.0
    for j in range(n):
        lhs += A[i, j] * x[j]
    assert abs(lhs - b[i]) < 1e-4
```

### Unifying Explicit and Implicit Integrators

$$
\left[\mathbf{I}-\beta \Delta t^{2} \mathbf{M}^{-1} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\left(\mathbf{x}_{t}\right)\right] \mathbf{v}_{t+1}=\mathbf{v}_{t}+\Delta t \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}_{t}\right)
$$

- $\beta = 0$, forward / semi-implicit Euler (Explicit)
- $\beta = 1/ 2$, middle point (Implicit)
- $\beta = 1$, backward Euler (Implicit) 

### Solve Faster

For millions of mass points and springs

- Sparse Matrices
- Conjugate Gradients
- Preconditioning
- Use Position-based Dynamics

## Lagrangian Fluid Simulation (SPH)

### Smoothed Particle Hydrodynamics

Use particles carrying samples of physical quntities, and a kernel $W$​, to approximate continuous fields ($A$ can be almost any spatially varying phisical attributes: density / pressure / ...)
$$
A(\mathbf{x})=\sum_{i} \cdot A_{i} \frac{m_{i}}{\rho_{i}} W\left(\left\|\mathbf{x}-\mathbf{x}_{j}\right\|_{2}, h\right)
$$
![image-20210724122234613](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210724122234613.png)

在 x 这点的物理场的值（即 $A(\mathbf{x})$​​​）等于核函数（如密度 / 压力等）加权平均了周围的粒子（本质是光滑了）的求和。核函数中间高四周低，使得更靠近该点影响更大，远离该点的影响变小（超过 support 半径 $h$​ 则为 0）

不需要 mesh，适合模拟自由表面 free-surface flows 流体（如水和空气接触的表面，反之，不适合烟雾（空间需要被填满）等），可以使用 “每个粒子就是以小包水” 理解

#### Equation of States (EOS) 

aka. Weakly Compressible SPH (WCSPH)

**Momentum Equation** ($\rho$ - density ($\rho_0$ - ideal), $B$ - bulk modulus, $\gamma$ - constant (usually ~ 7) )

(Actually the Navier-Stoke's without considering viscosity)
$$
\begin{gathered}
\frac{D \mathbf{v}}{D t}=-\frac{1}{\rho} \nabla p+\mathbf{g} , \quad p=B\left(\left(\frac{\rho}{\rho_{0}}\right)^{\gamma}-1\right) \\
A(\mathbf{x})=\sum_{i} A_{i} \frac{m_{i}}{\rho_{i}} W\left(\left\|\mathbf{x}-\mathbf{x}_{j}\right\|_{2}, h\right), \quad \rho_{i}=\sum_{j} m_{j} W\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}, h\right)
\end{gathered}
$$
$\rho_i$​ 通过质量加权, $D$ - Lagrange derivative

#### Gradient in SPH

$$
\begin{gathered}
A(\mathbf{x})=\sum_{i} A_{i} \frac{m_{i}}{\rho_{i}} W\left(\left\|\mathbf{x}-\mathbf{x}_{j}\right\|_{2}, h\right) \\
\nabla A_{i}=\rho_{i} \sum_{j} m_{j}\left(\frac{A_{i}}{\rho_{i}^{2}}+\frac{A_{j}}{\rho_{j}^{2}}\right) \nabla_{\mathbf{x}_{i}} W\left(\left\|\mathbf{x}_{i}-\mathbf{x}_{j}\right\|_{2}, h\right)
\end{gathered}
$$

Not really accurate but at least symmetric and momentum conserving (to add viscosity etc. Laplacian should be introduced)

#### SPH Simulation Cycle

- For each particle $i$, compute $\rho_i = \sum_j m_j W(||\mathbf{x}_i - \mathbf{x}_j||_2,h)$

- For each particle $i$, compute $\nabla p_i$ using the gradient operator

- Symplectic Euler steps (similar as the mass-spring model)
  $$
  \mathbf{v}_{t+1} = \mathbf{v}_t + \Delta t\frac{D\mathbf{v}}{Dt}, \quad \mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \frac{D\mathbf{v}}{Dt}
  $$

#### Variant of SPH

- Predictive-Corrective Impcompressible SPH (PCI-SPH) - 无散，更接近不可压缩
- Position-based Fluids (PBF) (Demo:`ti eample pbf2d`) - Position-based dynamics + SPH
- Divergence-free SPH (DFSPH) - (velocity)

Paper: *SPH Fluids in Computer Graphics*, *Smooted Particle Hydrodynamics Techniques for Physics Based Simulation of Fluids and Solids* (2019)

### Courant-Friedrichs-Lewy (CFL) Conditions (Explicit)

One upper bound of time step size: (def. using velocity other than stiffness)
$$
C=\frac{u \Delta t}{\Delta x} \leq C_{\max } \sim 1
$$
($C$ - Courant number, CFL number, $u$ - maximum velocity, $\Delta t$ - time step, $\Delta x$ - length interval (e.g. particle radius and grid size))

**Application**: estimating allowed time step in (explicit) time integration.

SPH: ~ 0.4; MPM (Material Point Method): 0.3 ~ 1; FLIP fluid (smoke): 1 ~ 5+

### Accelerating SPH: Neighborhood Search

Build spatial data structure such as voxel grids $O(n^2) \rightarrow O(n)$

**Neighborhood search with hashing** 

精确找到距离不超过 $h $​​​​ 的粒子，每个 voxel 维护一个 list（所有位置在这个 voxel 里的粒子），每个需要查找的粒子先确定相应 voxel 再查找周围的 voxel（枚举所有包含的粒子，通常是常数）- SPH 的瓶颈

![image-20210726000447459](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210726000447459.png) 

Ref: Compact hashing

### Other Paricle-based Simulation Methods

- Discrete Element Method (DEM) - 刚体小球，如沙子模拟
- Moving Particle Semi-implicit (MPS) - 增强 fluid incompressibility
- Power Particles - Incompressible fluid solver based on power diagrams
- A peridynamics perspective on spring-mass fracture  

## Exporting Results

Exporting Videos

- `ti.GUI.show`: save screenshots / `ti.imwrite(img, filename)`
- `ti video`: creates `video.mp4` (`-f 24` / `-f 60`)
- `ti gif -i input.mp4`: Convert `mp4` to `gif`



# Lecture 3  Lagrangian View (2)



弹性、有限元基础、Taichi 高级特性

## Elasticity

### Deformation

**Deformation Map $\phi$​​** (to describe): A (vector to vector) function that relates rest material position and deformed material position

$$\mathbf{x}_{\mathrm{deformed}} = \phi (\mathbf{x}_{\mathrm{rest}})$$

**Deformation gradient $\mathbf{F}$**:  $\mathbf{F} = \frac{\partial \mathbf{x}_{\mathrm{deformed}}}{\partial \mathbf{x}_{\mathrm{rest}}}$	(Translational invariant, same deformation gradients for $\phi_1$ and $\phi_1 + c$)

**Deform / rest volume ratio**:  $J = \mathrm{det} (\mathbf{F})$ 	（`F[None] = [[2, 0], [0, 1]]`（横向拉伸了））

### Elasticity

#### **Hyperelastic material**

Hyperelastic material: Stress-strain relationship is defined by a **strain energy density function**	$\psi = \psi(\mathbf{F})$

$\psi$​ is a potential function that penalizes deformation; $\mathbf{F}$ represents the strain (means deformation gradient)

#### **Stress Tensor**

- The First Piola-Kirchhoff stress tensor (PK1): $\mathbf{P(F)} = \frac{\partial \psi(\mathbf{F})}{\partial \mathbf{F}}$​ (easy to compute but in rest space, asymmetric)
- Kirchhoff stress: $\tau$
- Cauchy stress tensor: $\sigma$ (True stress, symmetric, conservation of angular momentum)

Relationship:  $\tau = j\sigma = \mathbf{PF}^T$;  $\mathbf{P} = J\sigma \mathbf{F}^{-T}$;  Traction  $t = \sigma^{T} \mathbf{n}$ （考虑的是截面的法向量，转置为 ^-T^）​

#### Elastic Moduli (Isotropic Materials)

- Young’s Modulus:  $E = \frac{\sigma }{\epsilon}$
- Bulk Modulus:  $K = -V \frac{dP}{dV}$ （常用与可压缩液体）
- Poisson’s Ratio:  $\nu \in [0.0, 0.5]$ （泊松比越大，越接近于不可压缩；Auxetics 材料可以拥有负泊松比，施加拉力材料反而变粗）

Lame parameters

- Lame’s first parameter:  $\mu$
- Lame’s second parameter (shear modulus / $G$):  $\lambda$​

Conversions:（通常指定 Young’s Modulus & Poisson’s Ratio，或 $E$ 和 $\nu$）$K = \frac{E}{3(1-2\nu)}$ ;  $\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}$ ;  $\mu = \frac{E}{2(1+\nu)}$

#### Hyperelastic Material Models

Popular in graphics:

- Linear elasticity (small deformation only, not consistence in rotation) - linear equation systems
- **Neo-Hookean**: (Commonly used in isotropic materials)
  - $\psi(\mathbf{F})=\frac{\mu}{2} \sum_{i}\left[\left(\mathbf{F}^{T} \mathbf{F}\right)_{i i}-1\right]-\mu \log (J)+\frac{\lambda}{2} \log ^{2}(J)$
  - $\mathbf{P}(\mathbf{F})=\frac{\partial \psi}{\partial \mathbf{F}}=\mu\left(\mathbf{F}-\mathbf{F}^{T}\right)+\lambda \log (J) \mathbf{F}^{-T}$
- (Fixed) Corotated:
  - $\psi(\mathbf{F})=\mu \sum_{i}\left(\sigma_{i}-1\right)^{2}+\frac{\lambda}{2}(J-1)^{2}$	($\sigma_i$ are singlular values of $\mathbf{F}$)
  - $\mathbf{P}(\mathbf{F})=\frac{\partial \psi}{\partial \mathbf{F}}=2 \mu(\mathbf{F}-\mathbf{R})+\lambda(J-1) J \mathbf{F}^{-T}$

## Lagrangian Finite Elements on Linear Tetrahedral Meshes

### The Finite Element Method (FEM)

Discretization sheme to build discrete equations using weak formulations of continuous PDEs

### Linear Tetrahedral (Triangular) FEM

Assumption: The deformation map $\phi$​ is affine and thereby the deformation gradient $\mathbf{F}$​ is constant

$\mathbf{x}_{\mathrm{deformed}} = \mathbf{F}(\mathbf{x}_{\mathrm{rest}}) + \mathbf{b}$	（$\mathbf{b}$ - offset，平移产生的形不产生弹性，不影响 gradient）

For every element $e$​, the elastic potential energy  $U(e) = \int_e\psi (\mathbf{F(x)}) d\mathbf{x} = V_e \psi (\mathbf{F} _e)$​​ （对每个元素这么设定，计算其弹性势能）

**Computing $\mathbf{F}_e$​** (for 2D triangles)

- Vertices: $\mathbf{a}_{\mathrm{rest}}$​; Deformed positions: $\mathbf{a}_\mathrm{deformed}$​ (same as $\mathbf{b}$​ and $\mathbf{c}$​)

- $ \mathbf{a}_\mathrm{deformed} = \mathbf{Fa}_\mathrm{deformed} + \mathbf{P}$ (same as $\mathbf{b}$ and $\mathbf{c}$​)

- Eliminate $\mathbf{P}$:  $\left(\mathrm{a}_{\text {deformed}}-\mathbf{c}_{\text {deformed}}\right)=\mathrm{F}\left(\mathrm{a}_{\text {rest}}-\mathrm{c}_{\text {rest}}\right)$  $\left(\mathrm{b}_{\text {deformed}}-\mathbf{c}_{\text {deformed}}\right)=\mathrm{F}\left(\mathrm{b}_{\text {rest}}-\mathrm{c}_{\text {rest}}\right)$

- $\mathbf{F}_{(2\times 2)}$ has 4 linear constrains (equations)

  $\mathbf{B} = [\mathbf{a}_\mathrm{rest} - \mathbf{c}_\mathrm{rest} | \mathbf{b}_\mathrm{rest} - \mathbf{c}_\mathrm{rest}]^{-1}$ ;  $\mathbf{D} = [\mathbf{a}_\mathrm{deformed} - \mathbf{c}_\mathrm{deformed} | \mathbf{b}_\mathrm{deformed} - \mathbf{c}_\mathrm{deformed}]$ ;  $\mathbf{F = DB}$​

  $\mathbf{B}$ is constant throughout the process, should be pre-computed

#### Explicit Linear Triangular (FEM) Simulation

Semi-implicit Euler time integration scheme:

$ \mathbf{v}_{t+1, i} =\mathbf{v}_{t, i}+\Delta t \frac{\mathbf{f}_{t, i}}{m_{i}} \quad \mathbf{x}_{t+1, i} =\mathbf{x}_{t, i}+\Delta t \mathbf{v}_{t+1, i} $​

$\mathbf{x}_{t,i}$​ and $\mathbf{v}_{t,i}$​​ are stored on the **vertices** of finite elements (triangles / tetrahedrons), $V_e$ - constant volumes

$\mathbf{f}_{t, i}=-\frac{\partial U}{\partial \mathbf{x}_{i}}=-\sum_{e} \frac{\partial U(e)}{\partial \mathbf{x}_{i}}=-\sum_{e} V_{e} \frac{\partial \psi\left(\mathbf{F}_{e}\right)}{\partial \mathbf{F}_{e}} \frac{\partial \mathbf{F}_{e}}{\partial_{ \mathbf{x}_{i}}}=-\sum_{e} V_{e} \mathbf{P}\left(\mathbf{F}_{e}\right) \frac{\partial \mathbf{F}_{e}}{\partial \mathbf{x}_{i}}$

Taichi’s AutoDiff system can use to compute $\mathbf{p}(\mathbf{F}_e)$​

``` python
for s in range(30):
    with ti.Tape(total_energy):  # Automatically diff. and write into a 'tape' and use in semi-implicit Euler
```

#### Implicit Linear Triangular FEM Simulation

Backward Euler Time Integration:  $\left[\mathbf{I}-\beta \Delta t^{2} \mathbf{M}^{-1} \frac{\partial \mathbf{f}}{\partial \mathbf{x}}\left(\mathbf{x}_{t}\right)\right] \mathbf{v}_{t+1}=\mathbf{v}_{t}+\Delta t \mathbf{M}^{-1} \mathbf{f}\left(\mathbf{x}_{t}\right)$

($\mathbf{M}$ - diagonal matrix to record the mass)

Compute force differentials: $\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = -\frac{\partial ^2 \psi}{\partial \mathbf{x}^2}$ (second order DE are not available in taichi)

## Higher Level of Taichi

### Objective Data-Oriented Programming

OOP -> **Extensibility** / **Maintainability**

An “object” contains its own **data** (py-var. / `ti.field`) and **method** (py-func. / `@ti.func` / `@ti.kernel`)

![OOP](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/OOP.svg) 

#### Python OOP in a Nutshell

``` python
class Wheel:	# def a class of 'wheel'
	def __init__(self, radius, width, rolling_fric):	# data, for a wheel: radius / width / fric
		self.radius = radius	# convert all data as `self.` (inside member of 'self')
		self.width = width
		self.rolling_fric = rolling_fric
        
	def Roll(self):	# Method of the 'wheel'
		...
         
w1 = Wheel(5, 1, 0.1)	# Instantiated Objects (different wheels)
w2 = Wheel(5, 1, 0.1)
w3 = Wheel(6, 1.2, 0.15)
w4 = Wheel(6, 1.2, 0.15)
```

If want to add some features, this method can inherit all past features and add on them (easy to maintain and extent)

#### Using Classes in Taichi

Hybrid scheme: Objective **Data**-Oriented Programming (ODOP)

​	-> More data-oriented -> usually use more vectors / matrices in the class

Important Decorators:

- `@ti.data_oriented` to decorate `class`
- `@ti.kernel` to decorate class members functions that are Taichi kernels
- `@ti.func` to decorate class members functions that are Taichi functions

**Caution**: if the variable is passed from Python scope, the `self.xxx` will still regard as a constant

**Features**:

- **Encapsulation**: Different classes can be stored in various `.py` scripts and can be called using `from [filename] import [classname]` in the `main` script.
- **Inheritance**: A @data_oriented class can inherit from another @data_oriented class. Both the data and methods are
  inherited from the base class.
- **Polymorphism**: Define methods in the child class that have the same name as the methods in the parent class. Proper methods will be called according to the instantiated objects.

Demo: `ti example odop_solar`:  $\mathbf{a} = GM\mathbf{r}/||\mathbf{r}||^3_2$ 

```python
import taichi as ti

@ti.data_oriented
class SolarSystem:
    def __init__(self, n, dt):	# n - planet number; dt - time step size
        self.n = n
        self.dt = dt
        self.x = ti.Vector(2, dt=ti.f32, shape=n)
        self.v = ti.Vector(2, dt=ti.f32, shape=n)
        self.center = ti.Vector(2, dt=ti.f32, shape=())
        
    @staticmethod	# @ti.func 还可以被额外的 @staticmethod（静态函数，同 py）修饰
    @ti.func	# 生成一个随机数
    def random_around(center, radius):	# random number in [center - radius, center + radius]
        return center + radius * (ti.random() - 0.5)  * 2
    
    @ti.kernel
    def initialize(self):	# initialize all the tensors
        for i in range (self.n):
            offset = ti.Vector([0.0, self.random_around(0.3, 0.15)])
            self.x[i] = self.center[None] + offset
            self.v[i] = [-offset[1], offset[0]]
            self.v[i] *= 1.5 / offset.norm()
            
    @ti.func	# still in class
    def gravity(self, pos):
        offset = -(pos - self.center[None])
        return offset / offset.norm()**3
    
    @ti.kernel	# sympletic Euler
    def integrate(self):
        for i in range(self.n):
            self.v[i] += self.dt * self.gravity(self.x[i])
            self.x[i] += self.dt * self.v[i]
            
solar = SolarSystem(9, 0.0005)	# 9 for n (planet number); 0.0005 for dt
solar.center[None] = [0.5, 0.5]
solar.initialize()

gui = ti.GUI("Solar System", background_color = 0x25A6D9)

while True:
    if gui.get_event():
        if gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
            solar.initialize()
    for i in range(10):
        solar.integrate()
    gui.circle([0.5, 0.5], radius = 20, color = 0x8C274C)
    gui.circles(solar.x.to_numpy(), radius = 5, color = 0xFFFFFF)
    gui.show()
```

### Metaprogramming  元编程

Allow to pass almost anything (including tensors) to Taichi kernels; Improve run-time performance by moving run-time costs to compile time; Achieve dimensionality independence (2D / 3D simulation codes simultaneously); etc. 很多计算可以在编译时间完成而非运行时间完成 (kernels are lazily instantiated)

Metaprogramming -> **Reusability**: 

​	Programming technique to generate other programs as the program’s data

![Metaprogramming](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/Metaprogramming.svg)

In Taichi the “Codes to write” section is actually `ti.templates` and `ti.statics`

#### Template

Allow to pass *anything* supported by Taichi (if directly pass something like `a = [43, 3.14]` (python list) -> error; need to modify as Taichi’s types `a = ti.Vector([43, 3.14])`)

- Primary types: `ti.f32`, `ti.i32`, `ti.f64`
- Compound Taichi types: `ti.Vector()`, `ti.Matrix()`
- Taichi fields: `ti.field()`, `ti.Vector.field()`, `ti.Matrix.field()`, `ti.Struct.field()`
- Taichi classes: `@ti.data_oriented`

Taichi kernels are instantiated whenever seeing a new parameter (even same typed)  

​	frequently used of templates will cause higher time costs

```python
@ti.kernel
# when calling this kernel, pass these 2 1D tensors (x, y) into template arguments (better to have same magnitude)
def copy(x: ti.template(), y: ti.template(), c: ti.f32): 	# template => to deliver fields
    for i in x:
        y[i] = x[i] + c 	# OK if the shape is the same (regardless the size)
```

**Pass-by-reference**: 

- Computations in the Taichi scope can NOT modify Python scope data (field is global, OK to modify)
- Computations in the Taichi scope can modify Taichi scope data (stored in the same RAM location)

``` python
vec = ti.Vector([0.0, 0.0])

@ti.kernel
def my_kernel(x: ti.template()):
    # x[0] = 2.0 is bad assignment, x is in py-scope (cannot modify a python-scope data in ti-scope)
    vec2 = x 
    vec2[0] = 2.0	# modify succeed, inside the ti-scope
    print(vec2)		# [2.0, 0.0]
    
my_kernel(vec)
```

The template initialization process could cause high overhead

``` python
import taichi as ti
ti.init()

@ti.kernel	# use template in this first kernel "hello"
def hello(i: ti.template()):
    print(i)
    
for i in range(100):
    hello(i)	# 100 different kernels will be created (if repeat this part, these 100 kernels will be reused)

@ti.kernel	# use int.32 argument for i in this second kernel "world"
def world(i: ti.i32):	
    print(i)
    
for i in range(100):
    world(i)	# The only instance will be reused (better in compiling)
```

##### Dimensionlity-independent Programming

``` python
@ti.kernel
def copy(x: ti.template(), y: ti.template()): 
    for I in ti.grouped(y):		# packing all y's index (I - n-D tensor)
    	# I is a vector with dimensionality same to y
		# If y is 0D, then I = ti.Vector([]), which is equivalent to `None` used in x[I]
		# If y is 1D, then I = ti.Vector([i])
		# If y is 2D, then I = ti.Vector([i, j])
		# If y is 3D, then I = ti.Vector([i, j, k])
		# ...
        x[I] = y[I]		# work for different dimension tensor
        
@ti.kernel
def array_op(x: ti.template(), y: ti.template()):
    for I in ti.grouped(x):
        # I is a vector of size x.dim() and data type i32
        y[I] = I[0] + I[1]  # = (i + j)
    # If tensor x is 2D, the above is equivalent to 
    for i, j in x:
        y[i, j] = i + j
```

##### Tensor-size Reflection

Fetch tensor dimensionlity info as compile time constants:

``` python
import taichi as ti

tensor = ti.var(ti.f32, shape = (4, 8, 16, 32, 64))

@ti.kernel
def print_tensor_size(x: ti.template()):
    print(x.dim())
    for i in ti.static(range(x.dim())):
        print(x.shape()[i])
        
print_tensor_size(tensor)	# check tensor size
```

#### Static

##### Compile-time Branching

Using compile-time evaluation will allow certain computations to happen when kernel are being instantiated. Saves the overhead of the computations at runtime (C++17: `if constexpr`) 

``` python
enable_projection = True

@ti.kernel
def static():	# branching process in compiler (No runtime overhead)
    if ti.static(enable_projection):	# (in this "if" branching condition)
        x[0] = 1
```

##### Forced Loop-unrolling

Use `ti.static(range(...))` : Unroll the loops at compile time（强制循环展开）

**Reduce** the loop **overhead** itself; loop over vector / matrix elements (in Taichi matrices must be compile-time constants)

``` python
import taichi as ti

ti.init()
x = ti.Vector(3, dt=ti.i32, shape=16)

@ti.kernel
def fill():
    for i in x:
        for j in ti.static(range(3)):
            x[i][j] = j
        print(x[i])

fill()
```

#### Variable Aliasing

Creating handy aliases for global var. and func. w/ cumbersome names to improve readability

``` python
@ti.kernel
def my_kernel():
    for i, j in tensor_a:	# 把 tensor a 的所有下标取出
        tensor b[i, j] = some_function(tensor_a[i, j])	# apply some func on all index in tensor a into tensor b
```

``` python
@ti.kernel
def my_kernel():
    a, b, fun = ti.static(tensor_a, tensor_b, some_function)	# Variable aliasing (must use ti.static)
    for i, j in a:
        b[i, j] = fun(a[i, j])	# use aliasing to replace the long names
```

#### Metadata

Data generated data (usually used to check whether the shape / size is the same or not for copy)

- **Field**: 

  - `field.dtype`: type of a field
  - `field.shape`: shape of a field

  ``` python
  import taichi as ti
  ti.init(arch = ti.cpu, debug = True)
  
  @ti.kernel
  def copy(src: ti.template(), dst: ti.template()):
      assert src.shape == dst.shape	# only executed when `debug = True`
      for i in dst:
          dst[i] = src[i]
          
  a = ti.field(ti.f32, 4)
  b = ti.field(ti.f32, 100)
  copy(a, b)
  ```

- **Matrix / Vector**:

  - `matrix.n`: rows of a mat
  - `matrix.m`: cols of a mat / vec

  ``` python
  @ti.kernel
  def foo():
  	matrix = ti.Matrix([[1, 2], [3, 4], [5, 6]])
  	print(matrix.n) # 3
  	print(matrix.m) # 2
  	vector = ti.Vector([7, 8, 9])
  	print(vector.n) # 3
  	print(vector.m) # 1
  ```

### Differentiable Programming

Forward programs evaluate $f(x)$, differentiable programs evaluates $\frac{\partial f(x)}{\partial x}$

Taichi supports **reverse-mode automatic differentiation (AutoDiff)** that back-propagates gradients w.r.t. a scalar (loss) function $f(x)$ （关于每个元素的导数）

- Use **Taichi’s tape** (`ti.Tape(loss)`) for both foward and gradient evaluation
- Explicitly use **gradient kernels** for gradient evaluation with more controls

#### Gradient-based Optimization

$$
\min _{\mathbf{x}} L(\mathbf{x}) = \frac{1}{2} \sum^{n-1}_{i=0} (\mathbf{x}_i - \mathbf{y} _ i)^2
$$

- Allocating tensors with gradients	（对 x 的导数 即为 x）

  ``` python
   x = ti.var(dt=ti.f32, shape=n, needs_grad=True)	# needs_grad=True: compute gradients 
  ```

- Defining loss function kernel(s):

  ``` python
  @ti.kernel
  def reduce():
      for i in range(n):
          L[None] += 0.5 * (x[i] - y[i]) ** 2		# compute the cummulative L(x) provided (+= atomic)
  ```

- Compute loss `with ti.Tape(loss = L): reduce()` (forward)

- Gradient descent: `for i in x: x[i] -= x.grad[i] * 0.1` (backward, auto)

Results: Loss exp. decreases to near 0

(also use for gradient descend method for fixing curves, FEM and MPM (with bg. grids to deal with self collision))

#### Application 1: Forces from Potential Energy Gradients

$$
\mathbf{f}_i = -\frac{\partial \phi(\mathbf{x})}{\partial \mathbf{x}_i}
$$

- Allocate a 0-D tensor to store potential energy	`potential = ti.var(ti.f32, shape=())`
- Define forward kernels from `x[i]`
- In a `ti.Tape(loss=potential)`, call the forward kernels
- Force on each particle is `-x.grad[i]`

(Demo: `ti example mpm_lagrangian_forces`)

#### Application 2: Differentiating a Whole Physical Process

Use AutoDiff for the whole physical process derivative

Not used for basic differentiation but optimization for **initial conditions**（初始状态的优化 / 反向模拟）/ **controller**

Need to record all tensors in the whole timestep in `ti.Tape()` ~ Requires high memory (for 1D tensor needs to use 2D Tape)

~ Use checkpointing to reduce memory consumption

### Visualization 

#### Print

- Random order

  Due to the parallel computations in `@ti.kernel` especially using GPU, the computations will be randomly done. If print in the for-loop in Taichi kernel, the results may be random.

- The `print()` in GPU is not likely to show until seeing `ti.sync()`

  ``` python
  @ti.kernel
  def kern():
  	print('inside kernel')
  
  print('before kernel')	# of course the first print
  kern()	# this 'inside kernel' may or may not be printed between these 2 prints
  print('after kernel')	
  ti.sync()	# force sync, 'after sync' will be the last print 100%
  print('after sync')
  ```

#### Visualizing 2D Results

Apply Taichi’s GUI system:

- **Set the window**: `gui = ti.GUI("Taichi MLS-MPM-128", res = 512, background_color = 0x112F41)`

- **Paint on the window**: `gui.set_image()` (especially for ray tracing or other…)

- **Elements**: `gui.circle/gui.circles(x.to_numpy(), radius = 1.5, color = colors.to_numpy())`

  `gui.lines()`, `gui.triangle()`, `gui.rects()`, …

- **Widgets**: `gui.button()`, `gui.slider()`, `gui.text()`, …

- **Events**: `gui.get_events()`, `gui.get_key_event()`, `gui.running()`, … (get keyboard / mouse /… actions)

#### Visualizing 3D Results

- **Offline**: Exporting 3D particles and meshes using `ti.PLYWriter`（输出二进制 ply 格式）

  Demo: `ti example export_ply/export_mesh`

  Houdini / Blender could be used to open (File - Import - Geometry in Houdini)

- **Realtime** (GPU backend only, WIP): `GGUI` (still in progress)



# Lecture 4  Eulerian View (Fluid Simulation)



回答问题：介质流过的速度

## Overview

### Material Derivatives

Connection of Lagrangian and Eulerian

(Use **D** for material derivatives. $\frac{\partial}{\partial t}$​ - Eulerian part, no spatial velocity, varies with time; $\mathbf{u}\cdot \nabla$​ - material movement, change by velocity part in scalar field (u - material (fluid) velocity / advective / Lag. / particle derivative))
$$
\frac{\mathrm{D}}{\mathrm{D}t} : = \frac{\partial}{\partial t} + \mathbf{u}\cdot \nabla
$$
For example:
$$
\begin{aligned} 
\text{Temperature:  } & \frac{\mathrm{D}T}{\mathrm{D}t} = \frac{\partial T}{\partial t} + \mathbf{u}\cdot \nabla T\\
\text{x-Velocity:  } & \frac{\mathrm{D}\mathbf{u}_x}{\mathrm{D}t}  = \frac{\partial \mathbf{u}_x}{\partial t} + \mathbf{u}\cdot \nabla \mathbf{u}_x
\end{aligned}
$$

### (Incompressible) Navier-Stokes Equations

$$
\rho \frac{\mathrm{D} \mathbf{u}}{\mathrm{D} t}=-\nabla p+\mu \nabla^{2} \mathbf{u}+\rho \mathrm{g} \quad \text{or} \quad\frac{\mathrm{D} \mathbf{u}}{\mathrm{D} t}=-\frac{1}{\rho} \nabla p+\nu \nabla^{2} \mathbf{u}+\mathbf{g}\,;\quad \nabla\cdot \mathbf{u} = 0
$$

Usually in graphics want low viscosity (delete the viscosity part) except for high viscous materials (e.g., honey)
$$
\Rightarrow \frac{\mathrm{D} \mathbf{u}}{\mathrm{D} t}=-\frac{1}{\rho} \nabla p + \mathbf{g}\,;\quad \nabla\cdot \mathbf{u} = 0
$$

### Operator Splitting

Split the equation (PDEs with time) into 3 parts: ($\alpha $ - any physical property (temperature / color / smoke density / ...))
$$ {\}
\begin{aligned}
\frac{\mathrm{D} \mathbf{u}}{\mathrm{D}t} &= \bold{0},\quad \frac{\mathrm{D} \mathbf{\alpha}}{\mathrm{D}t} = \bold{0}\quad\text{(advection)}\\

\frac{\partial \mathbf{u}}{\partial t} &= \mathbf{g} \quad\text{(external forces, optional)}\\
\frac{\partial \mathbf{u}}{\partial t} &= -\frac{1}{\rho} \nabla p \quad \text{s.t.} \quad \nabla \cdot \mathbf{u} = \bold{0} \quad \text{(projection)}
\end{aligned}
$$

### Time Discretization

(for each time step using the splitting above)

- **Advection**: “Move” the fluid field (no external forces, just the simple version), solve $\mathbf{u}^*$ using $\mathbf{u}^t$​
- **External Forces** (usually gravity accelaration, optional): evaluate $\mathbf{u}^{**}$ using $\mathbf{u}^{*}$​
- **Projection**: make velocity field $\mathbf{u}^{t+1}$ divergence-free based on $\mathbf{u}^{**}$​ (adding pressure)

## Grid

### Spatial Discretization 

#### Using Cell-centered Grids 

(evenly distributed)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210805202511587.png" alt="image-20210805202511587" style="zoom:33%;" /> 

$\mathbf{u}_x$, $\mathbf{u}_y$ and $p$​ are stored at the center of the cells (horizontal and vertical offset = 0.5 cell)

``` python
n, m = 3, 3
u = ti.var(ti.f32, shape = (n, m))	# x-comp of velocity
v = ti.var(ti.f32, shape = (n, m))	# y-comp of velocity
p = ti.var(ti.f32, shape = (n, m))	# pressure
```

#### Using Staggered grids

Stored in various location (Red - $\mathbf{u}_x$; Green: $\mathbf{u}_y$; Blue - $p$)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210805202532004.png" alt="image-20210805202532004" style="zoom:33%;" /> Red: 3x4; Green: 4x3; Blue: 3x3

``` python
n, m = 3, 3
u = ti.var(ti.f32, shape = (n+1, m))	# x-comp of velocity
v = ti.var(ti.f32, shape = (n, m+1))	# y-comp of velocity
p = ti.var(ti.f32, shape = (n, m))		# pressure
```

### Bilinear Interpolation

Interpolate $(x,y)$ with the 4 points by weighted average

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210805233100492.png" alt="image-20210805233100492" style="zoom:50%;" /> 

## Advection

Different **Schemes**: Trade-off between numerical viscosity, stability, performance and complexity

- Semi-Lagrangian Advection
- MacCormack / BFECC
- BiMocq^2^
- article Advection (PIC / FLIP / APIC / PolyPIC)

### Semi-Lagrangian Advection

#### Scheme

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210805233608015.png" alt="image-20210805233608015" style="zoom:33%;" /> 

Velocity field constant, for very short time step $\Delta t$​, velocity assumed to be constant (p -> x)

``` python
@ti.func
def semi_lagrangian(x, new_x, dt):
    for I in ti.grouped(x):	# loop over all subscripts in x
        new_x[I] = sample_bilinear(x, backtrace(I, dt))	# backtrace means the prev. value of x at the prev. dt
        # (find the position of the prev. x value in this field, and do bilinear interpolation and give to new_x)
```

#### Problem

velocity field not constant! -> Cause magnification / smaller / blur

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210805234531975.png" alt="image-20210805234531975" style="zoom:33%;" /> 

The real trajectory of material parcels can be complex (Red: a naive estimation of last position; Light gree: the true previous position.)

#### Solutions

(-> initial value problem for ODE, simply use the naive algorithm = forward Euler (RK1); to solve the problem can use RK2 scheme)

- Forward Euler (RK1) 

  ``` python
  p -= dt * velocity(p)
  ```

- Explicit Midpoint (RK2)

  ``` python
  p_mid = p - 0.5 * dt * velocity(p)
  p -= dt * velocity(p_mid)
  ```

  (Blur but no become smaller, usually enough for most computations)

- RK3 (weighted average of 3 points)

  ``` python
  v1 = velocity(p)
  p1 = p - 0.5 * dt * v1
  v2 = velocity(p1)
  p2 = p - 0.75 * dt * v2
  v3 = velocity(p2)
  p -= dt * (2/9 * v1 + 1/3 * v2 + 4/9 * v3)
  ```

  (Result similar to RK2)

~ Blur: Usually because of the use of bilinear interpolation (numerical viscosity / diffusion) (causes energy reduction) -> BFECC

### BFECC / MacCormack 

#### Scheme

BFECC: Back and Forth Error Compensation and Correction (Good reduction of blur especially for static region boundaries)

- $\mathbf{x}^* = \text{SL} (\mathbf{x}, \Delta t)$​​ : Use Semi-Lagrangian 1 time for a result of x (as x*) for 1 dt forward
- $\mathbf{x}^{**} = \text{SL} (\mathbf{x}^*, -\Delta t)$​​​ : 1 dt​​ backward (x** supposed to be equal to the original x (for ideal advection) -> advection error)
- Error Estimation: $\mathbf{x}^{\text{error}} = 0.5 (\mathbf{x}^{**} - \mathbf{x})$
- Apply the error (correction): $x^{\text{final}} = \mathbf{x}^* + \mathbf{x}^{\text{error}}$ (May cause overshooting / artifacts)

``` python
@ti.func
def maccormack(x, dt):	# new_x = x*; new_x_aux = x**; 
    semi_lagrangian(x, new_x, dt)	# step 1 (forward dt)
    semi_lagrangian(new_x, new_x_aux, -dt)	# step 2 (backward dt)
    
    for I in ti.grounped(x):
        new_x[I] = new_x[I] + 0.5 * (x[I] - new_x_aux[I])	# Error estimation (new_x = x^{final})
```

#### Problem: Overshooting

~ Gibbs Phenomen at boundary because of this correction: `0.5 * (x[I] - new_x_aux[I])`

Idea: Introduce a clipping function

``` python
....

  for I in ti.grounped(x):
        new_x[I] = new_x[I] + 0.5 * (x[I] - new_x_aux[I])	# prev. codes
        
        if ti.static(mc_clipping):
            source_pos = backtrace(I, dt)
            min_val = sample_min(x, source_pos)
            max_val = sample_max(x, source_pos)
            
            if new_x[I] < min_val or new_x[I] > max_val:
                new_x[I] = sample_bilinear(x, source_pos)
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210806002649157.png" alt="image-20210806002649157" style="zoom:44%;" /> (Artifacts)

## Chorin-Style Projection

To ensure the velocity field is divergence free after projection - <u>Constant Volume</u> (need linear solver - MGPCG)

Expand using finite difference in time ($\mathbf{u}^*$ - the projected velocity field, $\nabla\cdot \nabla p$ - The divergence degree of the gradient of the pressure)

> **Divergence** ($\mathrm{div} = \div$​​): the total generation / sink of the fluid (+ for out, - for in). For actual (imcompressible) fluid flow: $\div \mathbf{F} = 0$​​ $\cur​​
>
> **Curl** ($\mathrm{curl = \curl}$​): Clockwise - $\curl \mathbf{F} < 0$​; Counter clockwise - $\curl \mathbf{F} > 0$​
>
> (REF: https://www.youtube.com/watch?v=rB83DpBJQsE)

$$
\begin{aligned}
\mathbf{u}^{*}-\mathbf{u} &=-\frac{\Delta t}{\rho} \nabla p \quad \text { s.t. } \quad \nabla \cdot \mathbf{u}^{*}=0 \\
\mathbf{u}^{*} &=\mathbf{u}-\frac{\Delta t}{\rho} \nabla p \quad \text { s.t. } \quad \nabla \cdot \mathbf{u}^{*}=0 \\
\nabla \cdot \mathbf{u}^{*} &=\nabla \cdot\left(\mathbf{u}-\frac{\Delta t}{\rho} \nabla p\right) \\
0 &=\nabla \cdot \mathbf{u}-\frac{\Delta t}{\rho} \nabla \cdot \nabla p \\
\nabla \cdot \nabla p &=\frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}
\end{aligned}
$$
Want: find a $p$ for the last formula

### Poisson’s Equation

$$
\nabla \cdot \nabla p = f \quad \text{or} \quad \Delta p = f
$$

($\Delta$ here represents $\nabla ^2 =\nabla \cdot \nabla$​, as the **Laplace operator**)

If $f = 0$​​, the equation becomes **Laplace’s Equation** (in this course generally not 0)

 ### Spatial Discretization (2D)

Discretize on a 2D grid: (central differential, 5 points)
$$
\begin{aligned}
(\mathbf{AP})_{i,j} & = (\nabla \cdot \nabla p)_{i,j} = \frac{1}{\Delta x^2} (-4 p_{i,j} + p_{i+1,j} + p_{i-1,j} + p_{i,j-1} + p_{i,j+1})\\
\mathbf{b}_{i,j} & = (\frac{\rho }{\Delta t} \nabla \cdot \mathbf{u})_{i,j} = \frac{\rho}{\Delta t\Delta x} (\mathbf{u}^x_{i+1,j} - \mathbf{u}^x_{i,j} + \mathbf{u}^y_{i,j+1} - \mathbf{u}^y_{i,j})

\end{aligned}
$$
Linear System:	$\mathbf{A}_{nm\times nm} \mathbf{p}_{nm} =\mathbf{b}_{nm}$​​ (very high dimension (especially for 3D), but sparse)​

#### $\nabla \cdot u$​ 

Divergency of velocity（速度的散度）: Quantity of fluids flowing in / out: Flows in = Negative; Flows out = Positive

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210806104706358.png" alt="image-20210806104706358" style="zoom:43%;" />

Remind: Staggered Grid: x-componenets of u - vertical boundaries; y-comp: horizontal boundaries
$$
(\nabla \cdot \mathbf{u})_{i, j}=\left(\mathbf{u}_{i+1, j}^{x}-\mathbf{u}_{i, j}^{x}+\mathbf{u}_{i, j+1}^{y}-\mathbf{u}_{i, j}^{y}\right)
$$

#### $\nabla\cdot \nabla p$​

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210806105359859.png" alt="image-20210806105359859" style="zoom:43%;" />
$$
(\nabla \cdot \nabla p)_{i, j}=\frac{1}{\Delta x^{2}}\left(-4 p_{i, j}+p_{i+1, j}+p_{i-1, j}+p_{i, j-1}+p_{i, j+1)}\right.
$$

#### Boundary Conditions

Dirichlet and Neumann boundaries  

- Open (regard as air):  $p = 0$
- Regard as solid:  $-4p_{i,j} \rightarrow -3p_{i,j}$ 

## Solving Large-scale Linear Systems  

$\mathbf{Ax = b}$

- Direct solvers (e.g. PARDISO) - small linear system
- Iterative solvers:
  - Gauss-Seidel
  - (Damped) Jacobi
  - (Preconditioned) Krylov-subspace solvers (e.g. conjugate gradients)

(Good numerical solver are usually composed of different solvers)

### Matrix Storage

$\mathbf{A}$: often sparse, symmetric and positive-defined (SPD) (Actually 2-D arrays)

Store $\mathbf{A}$:

- As a dense matrix: e.g. `float A[1024][1024]` (doesn’t scale, but works for small matrices)
- As a sparse matrix: (various formats: CSR, COO, ..)
- Don’t store it at all (**Matrix-free**, often the ultimate solution) (Fetching values costs much, so in graphics usually not store)

### Krylov-subspace Solvers

- **Conjugate gradients (CG)** (Commonly used)
- Conjugate residuals (CR)
- Generalized minimal residual method (GMRES)
- Biconjugate gradient stabilized (BiCGStab)

#### Conjugate gradients

**Basic Algorithm**: (energy minimization)

$\mathbf{r}_{0}=\mathbf{b}-\mathbf{A} \mathbf{x}_{0}$	# r - Residual
$\mathbf{p}_{0}=\mathbf{r}_{0}$	# Estimation (x should be add a ? p to approach the solution)
$k=0$
while True:

​	$\alpha_{k}=\frac{\mathbf{r}_{k}^{\top} \mathbf{r}_{k}}{\mathbf{p}_{k}^{\top} \mathbf{A} \mathbf{p}_{k}}$

​	$\mathbf{x}_{k+1}=\mathbf{x}_{k}+\alpha_{k} \mathbf{p}_{k}$
​	$\mathbf{r}_{k+1}=\mathbf{r}_{k}-\alpha_{k} \mathbf{A} \mathbf{p}_{k}$
​	if $\left\|\mathbf{r}_{k+1}\right\|$ is sufficiently small, break 

​	$\beta_{k}=\frac{\mathbf{r}_{k+1}^{\top} \mathbf{r}_{k+1}}{\mathbf{r}_{k}^{\top} \mathbf{r}_{k}}$
​	$	\mathbf{p}_{k+1}=\mathbf{r}_{k+1}+\beta_{k} \mathbf{p}_{k}$
​	$k=k+1$
return $\mathbf{x}_{k+1}$

#### Eigenvalues and Condition Numbers

Convergence related to condition numbers

Remind:  

- if $\mathbf{Ax = \lambda x}$​​ : **Eigenvalue**: $\lambda$; **Eigenvector**: $\mathbf{x}$​

  即 $\mathbf{A}$ 作用于 $\mathbf{x}$ 使 $\mathbf{x}$ 方向不改变，大小缩放 $\lambda$ 倍

- **Condition Number $\kappa$** of SPD matrix $\mathbf{A}$:  $\kappa(\mathbf{A}) = \lambda_{\max} / \lambda_{\min}$

<u>A smaller condition number causes faster convergence</u>

#### Warm Starting

Starting with an closer initial guess results in fewer interations needed

Using $\mathbf{p}$​ from the last frame as the initial guess of the current frame (Jacobi / GS / CG work well but not good for MGPCG)

 #### Preconditioning

Find an approximate operator $\mathbf{M}$ that is close to $\mathbf{A}$ but easier to convert

$\mathbf{Ax=b\quad\Leftrightarrow \quad M^{-1}Ax = M^{-1}b}$

$\mathbf{M}^{-1}\mathbf{A}$ may have a smaller condition number

**Common Preconditioners:**

- Jacobi (diagonal) preconditioner $\mathbf{M} = \mathrm{diag}(\mathbf{A})$
- Poisson preconditioner
- (incomplete) Cholesky decomposition
- **Multigrid**: $\mathbf{M}$ = very complex linear operator that almost inverts $\mathbf{A}$​ (Commonly used in graphics)
- Fast multipole method (FMM)

##### Multigrid preconditioned conjugate gradients (MGPCG)  

Residual reduction very fast in several iterations

A. McAdams, E. Sifakis, and J. Teran (2010). “A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids.”. In: Symposium on Computer Animation, pp. 65–73.  

Taichi demo: 2D/3D multigrd: `ti example mgpcg_advanced` 



# Lecture 5  Poisson’s Equation and Fast Method



## Poisson’s Equation and its Fundamental Solution

Using the view of PDE
$$
\begin{aligned}
&\nabla^2 \phi = -\rho\\
&\phi (x=0) = 0\\
&\phi (x) = \int _{\Omega} \frac{\rho(y)}{4\pi ||x-y||_2} dy\\
&\phi_j = \sum^{N}_{j=1} \frac{m_j}{4\pi R_{ij}}
\end{aligned}
$$
For example: Gravitational Problem （势场满足密度的泊松方程）
$$
\begin{aligned}
f(x) &= \nabla \phi\\
f(x) &= -\sum^{N}_{j=1,j\neq i} \nabla x_i \left(\frac{\rho_j \nu_j}{4\pi ||x_i - x_j||_2} \right)\\
f_i &= -\sum^{N}_{j=1,j\neq i} \frac{\rho_j \nu_j}{4\pi ||x_i - x_j||_2^3} 
\end{aligned}
$$
If N particles in M points computaiton -> Required $O(NM)$

## Fast Summation

快速多级展开

### M2M Transform

- 2D (Using Complex number to represent in coordinate): $\phi(Z) = \mathrm{Re}(\sum_j q_j \log(Z- {Z}_j))$​​​ (in real part; log - Green Function in 2D)

- Consider a source and its potential:  ![image-20210813171147818](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210813171147818.png) 

  - Apply Taylor Expansion: $\phi(Z) = q_i \log (Z-\varepsilon_i) = q_i \log(Z) - \sum^{p}_{k=1} \frac{q_i\varepsilon_i^k}{kZ^k}$	（从 0 点带来的势 + 高阶项）

- Multipole Expansion:  ![image-20210813171435607](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210813171435607.png) (every point in the cloud represeted as q~j~)

  - （来自 0 点的势作用之和 + 高阶项对于 Z 的影响（由形状））
    $$
    \phi(Z) = \sum_j q_j\log (Z-Z_j) = (\sum_j q_j) \log(Z) - \sum^{P}_{k=1} \frac{\sum_j \frac{q_jZ_j^k}{k}}{Z^k}
    $$
    将公式抽象（$Q = \sum_j q_j$）后，高阶项会几何收敛
    $$
    \begin{aligned}
    Q(k) &= -\sum_j \frac{q_j Z_j^k}{k}\\
    \phi(Z) &= Q \log Z +\sum^P_{k=1} \frac{Q_k}{Z^k}
    \end{aligned}
    $$
    $O (N\log N)$​​ Algorithm: Tree code (Compute Q and Q~k~ for each level, the grids will be less and less)

- One step further: 

  ​		![image-20210813173831941](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210813173831941.png) 

  - Problem: If we know M-Expansion at (electron cloud) Z~1~ (M~1~ = {z~1~, Q, Q~k~}), what is the M-Expansion at Z~2~ (M~2~ = {z~2~, Q~2~, Q~2,k~})?

    Actually a M to M translation (from z~1~ to z~2~)

    We want to obtain the coefficients from M~1~ , not q~i~’s
    $$
    \phi(Z) = Q\log(Z-Z_2) +\sum^P_{k=1} \frac{b_k}{Z^k}
    $$
    Recall $\phi(Z) = Q\log Z + \sum^P_{k=1} \frac{Q_k}{Z^k}$ (where $Q$​ here is the potential of all electrons, geometric convergence)

  - $b_k$​​ is a generalization of $Q_k$​​
    $$
    b_k = -\frac{Q(Z_1 - Z_2)^k}{k } + \sum^{k}_{i=1}Q_i (Z_1 - Z_2)^{k-i} \begin{pmatrix}k-1 \\ i-1\end{pmatrix}
    $$

    $$
    \phi (Z) = Q\log(Z-C) + \sum^{P}_{k=1} \frac{b_k}{(Z-C)^k}
    $$

  - View Sources as Multipole: $Q = q_i$

    Reveal “Multipole Expansion”
    $$
    \phi (Z) = Q\log(Z-C) + \sum^{P}_{k=1} \frac{\sum_j \frac{-q_j(Z_j -C)^k}{k}}{(Z-C)^k}
    $$
    From “Multipoles” (every multipole - an electron) ~ M2M Transform

    Compute bk with “Rest of terms”

``` c
struct Multipole{
	vec2 center;	// central point (in coordinate)
	complex q[p];	// rest of terms (in complex)
};
//source charge is a special Multipole,
//with center = charge pos, q0 = qi, q1…q4 =0

Multipole M2M(std::vector<Multipole> &qlist)
{
	Multipole res;
	res.center = weightedAverageof(qlist[i].center*qlist[i].q[0]);	// new center point (weighted average)
	q[0] = sumof(qlist[i].q[0]);	// q0 terms 
    
    for(k=1:p){
        res.q[k]=0;
        for(j=0:qlist.size()) {
            res.q[k] += computeBk(qlist[i]);	// compute bk function
        }
    }
	return res;
}
```

(Electrons - Multipoles and Multipole - Multipole)

For M2M Transform: Both z~1~ and z~2~ are far from Z; For M2L (Local pole ~ interpolation) Transform: z~1~ near Z

### **M2L Transform**

If we know M-Expansion at c~1~ (M~1~ = {c~1~, Q, Q~k~}), what is the polynomial at z~1~, so that potentials at neighbor z can be evaluated.

![image-20210813233116830](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210813233116830.png)
$$
\begin{aligned}
\phi (Z) &= Q\log (Z-C) + \sum^{P}_{k=1} \frac{b_k}{(Z-C)^k}\\
&= Q\log (Z_1 - C + Z - Z_1) + \sum^{P}_{k=1} \frac{b_k}{(Z_1-C+Z-Z_1)^k}\\
&= \underbrace{Q\log (Z_1 - C) + \sum^{P}_{k=1} \frac{b_k}{(Z_1-C)^k} }_{\phi(Z_1) }+ \underbrace{\sum^{P}_{l=1} b_l(Z-Z_1)^l}_\mathrm{ H.O.T.}
 
\end{aligned}
$$
where H.O.T. is high order turbulence, from multipole to local pole
$$
b_l = -\frac{Q}{l(C-Z_1)^l} + \frac{1}{(C-Z_1)^l} \sum^{P}_{k=1} \frac{b_k}{(Z_1-C)^k} \begin{pmatrix} l+k-1 \\ k-1\end{pmatrix}
$$

``` c
struct Localpole{
    vec2 center;
    complex b[p];
};
```

### **L2L Transform**

If we know L-Expansion at c~1~ (L~1~ = {c~1~, B}), what is the polynomial at c~2~, so that potentials at neighbor z around c~2~ can be evaluated?	Honer Scheme

![image-20210813234942106](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20210813234942106.png)
$$
\sum_{k=0}^{P}a_k (Z-Z_0) ^k \longrightarrow \sum _{k=0}^P b_k (Z)^k
$$

### Comparison

- Multipole expansion: Coarsening
- Localpole expansion: Interpolation

$O(N)$ Alogarithm: $\sum_{l} C^P \frac{N}{4^l} = O(C^P N)$ 



---

26:00







# —

# Taichi Graphic Course S1 (NEW)

Lecture 0-2 is in the above notes



# Lecture 3 Advanced Data Layout (21.10.12)

## Advanced Dense Data Layouts

(`ti.field()` is dense and `@ti.kernel` is optimized for field => Data Oriented => focus on data-access / decouple the ds from computations)

CPU -> wall of computations; GPU / Parallel Frame -> wall of data access (memory access)

### Packed Mode

- Initiaized in `ti.init()`

  - default: `packed = False`: do padding

    `a = ti.field(ti.i32, shape=(18,65))	# padded to (32, 128)`

  - `packed = True`: for simplicity (No padding)

### Optimized for Data-access

- 1D fields:

  Prefetched (every time access the memory -> slow): align the access order with the memory order

- N-D fields: stored in our 1D memory: store as what it accesses

  Ideal memory layout of oan N-D field (not matter line / col. / block major)

  For example in C/C++:

  ``` c
  int x[3][2];	// row-major
  int y[2][3];	// col.-major (actually still 3x2)
  
  foo(){
      for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 2; j++) {
              d_somthing(x[i][j]);
          }
      }
      
      for (int j = 0; j < 2; j++) {
          for (int i = 0; i < 3; i++) {
              do_somthing(y[j][i]);
          }
      }
  }
  ```

### Upgrade `ti.field()`

From `shape` to `ti.root`

````python
x = ti.Vector.field(3, ti.f32, shape = 16)
````

change to:

````python
x = ti.Vector.field(3, ti.f32)
ti.root.dense(ti.i, 16).place(x)
````

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012192916022.png" alt="image-20211012192916022" style="zoom:70%;" />

One step futher:

```python
x = ti.field(ti.i32)
ti.root.dense(ti.i, 4).dense(ti.j, 4).place(x)
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012192924233.png" alt="image-20211012192924233" style="zoom:60%;" />

- SNode: Structure node

- An SNode Tree:

  ``` python
  ti.root		# the root of the SNode-tree
  	.dense()	# a dense container decribing shape
      	.place(ti.field())	# a field describing cell data
      
      ...
  ```

A Taichi script uses dense equiv. to the above C/C++ codes

```python
x = ti.field(ti.i32)
y = ti.field(ti.i32)
ti.root.dense(ti.i, 3).dense(ti.j, 2).place(x)	# row-major
ti.root.dense(ti.j, 2).dense(ti.i, 3).place(y)	# col.-major

@ti.kernel
def foo():
    for i, j in x:
        do_something(x[i, j])
        
	for i, j in y:
        do_something(y[i, j])
```

#### Row & Column Majored Fields

Col: <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012200823340.png" alt="image-20211012200823340" style="zoom:53%;" /> Row:<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012201031533.png" alt="image-20211012201031533" style="zoom:50%;" />

Access: struct for (the access order will altered for row and col. -majored)

**Example**: loop over `ti.j` first (row-majored)

``` python
import taichi as ti
ti.init(arch = ti.cpu, cpu_max_num_threads=1)

x = ti.field(ti.i32)
ti.root.dense(ti.i, 3).dense(ti.j, 2).place(x)
# row-major

@ti.kernel
def fill():
	for i,j in x:
		x[i, j] = i*10 + j
        
@ti.kernel
def print_field():
	for i,j in x:
		print("x[",i,",",j,"]=",x[i,j],sep='', end=' ')
        
fill()
print_field()       
```

#### Special Case (dense after dense)

##### Hierachical 1-D field (block storage)

- Access like a 1-D field
- Store like a 2-D field (in blocks)   

``` python
x = ti.field(ti.i32)
ti.root.dense(ti.i, 4).dense(ti.i, 4).place(x) # hierarchical 1-D
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012193228785.png" alt="image-20211012193228785" style="zoom:50%;" />

##### Block major

e.g. for 9-point stencil

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211019175844706.png" alt="image-20211019175844706" style="zoom:53%;" /> 

``` python
x = ti.field(ti.i32)
ti.root.dense(ti.ij, (2,2)).dense(ti.ij, (2,2)).place(x) # Block major hierarchical layout, size = 4x4
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012193352701.png" alt="image-20211012193352701" style="zoom: 67%;" />

Compare to flat layouts:

``` python
z = ti.field(ti.i32, shape = (4,4))
# row-majored flat layout; size = 4x4
```

#### Array of Structure (AoS) and Structure of Arrays (SoA)

``` c
struct S1 {
    int x[8];
    int y[8];        
} S1 soa;
```

``` c
struct S2 {
    int x;
    int y;        
} S2 aos[8];
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012193933727.png" alt="image-20211012193933727" style="zoom:50%;" />

**Switching**: `ti.root.dense(ti.i, 8).place(x,y)` -> `ti.root.dense(ti.i, 8).place(x)` + `ti.root.dense(ti.i, 8).place(y)  `

## Sparse Data Layouts

### SNode Tree (Extended)

- **root**
- **dense**: fixed length contiguous array
- **bitmasked**: similar to dense, but it also uses a mask to maintain sparsity info
- **pointer**: stores pointers instead of the whole structure to save memory and maintain sparsity

Dense SNode-Tree:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012194553239.png" alt="image-20211012194553239" style="zoom:67%;" />

### Pointer

But the space occupation rate could be low -> use pointer (when no data in the space a pointer points -> set to 0)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012194725213.png" alt="image-20211012194725213" style="zoom:67%;" />

#### Activation

Once writing an inactive cell: `x[0,0] = 1` (activate the whole block the pointer points to (cont. memory))

If print => inactivate will not access and return 0

#### Pointer in Pointer

Actually ok but can be a waste of memory (pointer = 64 bit / 8 bytes). can also break cont. in space

### Bitmasked

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012195127835.png" alt="image-20211012195127835" style="zoom:67%;" />

The access / call is similar to a dense block. Cost 1-bit-per-cell extra but can skip some …

### API

- Check activation status:

  `ti.is_active(snode, [i, j, ...])`

  e.g.: `ti.is_active(block1, [0])	# = True`

- Activate / Deactivate cells:

  `ti.activate/deactivate(snode, [i,j])`

- Deactivate a cell and its children:

  `snode.deactivate_all()`

- Compute the index of ancestor

  `ti.rescale_index(snode/field, ancestor_snode, index)`

  e.g.: `ti.rescale_index(block2, block1, [4])	# = 1`

### Put Together

Example

- A column-majored 2x4 2D sparse field:  

  ``` python
  x = ti.field(ti.i32)
  ti.root.pointer(ti.j,4).dense(ti.i,2).place(x)
  ```

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211012195409845.png" alt="image-20211012195409845" style="zoom:50%;" />

- A block-majored (block size = 3) 9x1 1D sparse field:  

  ``` python
  x = ti.field(ti.i32)
  ti.root.pointer(ti.i,3).bitmasked(ti.i,3).place(x)
  ```

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211019180223502.png" alt="image-20211019180223502" style="zoom:60%;" />



# Lecture 4 Sparse Matrix, Debugging and Code Optimization (21.10.19)

## Sparse Matrix and Sparse Linear Algebra

Naive Taichi Implementation: hard to maintain

### Build a Sparse Matrix

- Sparse Matrix Solver: `ti.SparseMatrixBuilder()` (Use triplet arrays to store the line / col / val)

  ``` python
  n = 4
  K = ti.SparseMatrixBuilder(n, n, max_num_triplets=100)
  
  @ti.kernel
  def fill(A: ti.sparse_matrix_builder()):
      for i in range(n):
          A[i, i] += 1
  ```

- Fill the builder with the matrices’ data: 

  ``` python
  fill(K)
  
  print(">>>> K.print_triplets()")
  K.print_triplets()
  # outputs:
  # >>>> K.print_triplets()
  # n=4, m=4, num_triplets=4 (max=100)(0, 0) val=1.0(1, 1) val=1.0(1, 2) val=1.0(3, 3) val=1.0
  ```

- Create sparse matrices from the builder

  ``` python
  A = K.build()
  print(">>>> A = K.build()")
  print(A)
  # outputs:
  # >>>> A = K.build()
  # [1, 0, 0, 0]
  # [0, 1, 0, 0]
  # [0, 0, 1, 0]
  # [0, 0, 0, 1]
  ```

### Sparse Matrix Operations

- Summation: `A + B`
- Subtraction: `A - B`
- Scalar Multiplication: `c * A` or `A * c`
- Element-wise Multiplication: `A * B`
- Matrix Multiplication: `A @ B`
- Matrix-vector multiplication: `A @ b`
- Transpose: `A.transpose()`
- Element Access: `A[i, j]`

### Sparse Linear Solver

``` python
# factorize
solver = ti.SparseSolver(solver_type="LLT")	# def.: LL (Lower Tri.) Transpose (2); also could be "LDLT" / "LU"
solver.analyze_pattern(A)	# pre-factorize
solver.factorize(A)		# factorization

# solve
x = solver.solve(b)	# (numpy) array

# check stats
# if not, could be the factorization error (symmetric? ..)
isSuccessful = solver.info()
print(">>>> Solve sparse linear systems Ax = b with solution x")
print(x)
print(f">>>> Computation was successful?: {isSuccessful}")
```

#### Example: Linear Solver（鸡兔同笼）

$$
\begin{bmatrix}
1 & 1 \\ 2 & 4
\end{bmatrix}
\begin{bmatrix}
^n \text{Chickens} \\ ^{n} \text {Rabbits} 
\end{bmatrix}
= 
\begin{bmatrix}
^n \text{Heads} \\ ^{n} \text {Legs} 
\end{bmatrix}
$$

Given the amount of heads and legs to compute the amount of chicken and rabbit

If the problem is solved intuitively as a linear system $\vb A x = b$, the time complexity will be at the level of $O(n^3)$ (even in sparse matrices)

-> Use `x = solver.solve(b)`

#### Example: Linear Solver (Diffusion)

$$
\frac{\partial T}{\partial t} = \kappa \grad^2 T\ ; \quad \grad^2T = \grad_{xx}^2T + \grad_{yy}^2T
$$

**Finite Differential Method with FTCS Scheme**

- **Spatially** discretized: 
  $$
  \frac{\partial T_{i, j}}{\partial t}=\frac{\kappa}{\Delta x^{2}}\left(-4 T_{i, j}+T_{i+1, j}+T_{i-1, j}+T_{i, j+1}+T_{i, j-1}\right)
  $$

- **Temporally** discretized: (explicit)
  $$
  \frac{T_{n+1}-T_n}{\Delta t} = \kappa\grad^2 T_n
  $$

``` python
@ti.kernel
def diffuse(dt: ti.f32):
	c = dt * k / dx**2
	for i,j in t_n:
		t_np1[i,j] = t_n[i,j]
		if i-1 >= 0:
			t_np1[i, j] += c * (t_n[i-1, j] - t_n[i, j])
		if i+1 < n:
			t_np1[i, j] += c * (t_n[i+1, j] - t_n[i, j])
		if j-1 >= 0:
			t_np1[i, j] += c * (t_n[i, j-1] - t_n[i, j])
		if j+1 < n:
			t_np1[i, j] += c * (t_n[i, j+1] - t_n[i, j])
```

**Matrix Representation**: (Explicit)
$$
\frac{T_{n+1}-T_{n}}{\Delta t}=\frac{\kappa}{\Delta x^{2}} \mathbf{D} T_{n} \Rightarrow T_{n+1}=\left(\mathbf{I}+\frac{\Delta t * \kappa}{\Delta x^{2}} \mathbf{D}\right) T_{n}
$$
$\kappa$ is the conductivity rate, higher $\kappa$ for higher propagation of the heat. But too high $\kappa$ can have wrong wrong results. (for example: the T~origin~ is 100 degree and $\kappa = 500$, the next step T will be -300 => oscillation results, unstable)
$$
\mathbf{D}=\left[\begin{array}{cccccccc}
-2 & +1 & & +1 & & & & & \\
+1 & -3 & +1 & & +1 & & & & \\
& +1 & -2 & & & +1 & & & \\
+1 & & & -3 & +1 & & +1 & & \\
& +1 & & +1 & -4 & +1 & & +1 & \\
& & +1 & & +1 & -3 & & & +1 \\
& & & +1 & & & -2 & +1 & \\
& & & & +1 & & +1 & -3 & +1 \\
& & & & & +1 & & +1 & -2
\end{array}\right]
$$
The temperature transfer function can be easier to express:

``` python
def diffuse(dt: ti.f32):
	c = dt * k / dx**2
	IpcD = I + c*D
	t_np1.from_numpy(IpcD@t_n.to_numpy())
	# t_np1 = t_n + c*D*t_n
                     
@ti.kernel
def fillDiffusionMatrixBuilder(A:
ti.sparse_matrix_builder()):
    for i,j in ti.ndrange(n, n):
        count = 0
		if i-1 >= 0:
			A[ind(i,j), ind(i-1,j)] += 1
			count += 1
		if i+1 < n:
			A[ind(i,j), ind(i+1,j)] += 1
			count += 1
		if j-1 >= 0:
			A[ind(i,j), ind(i,j-1)] += 1
			count += 1
		if j+1 < n:
			A[ind(i,j), ind(i,j+1)] += 1
			count += 1
		A[ind(i,j), ind(i,j)] += -count
```

**Matrix Representation**: (Implicit)
$$
\frac{T_{n+1}-T_{n}}{\Delta t}=\frac{\kappa}{\Delta x^{2}} \mathbf{D} T_{n+1} \Rightarrow T_{n+1}=\left(\mathbf{I}-\frac{\Delta t * \kappa}{\Delta x^{2}} \mathbf{D}\right)^{-1} T_{n}
$$

``` python
def diffuse(dt: ti.f32):
	c = dt * k / dx**2
	ImcD = I - c*D
    
	# linear solve: factorize
	solver = ti.SparseSolver(solver_type="LLT")
	solver.analyze_pattern(ImcD)
	solver.factorize(ImcD)
    
	# linear solve: solve
	t_np1.from_numpy(solver.solve(t_n))
	# t_np1 = t_n + c*D*t_np1
    
@ti.kernel
def fillDiffusionMatrixBuilder(A:
ti.sparse_matrix_builder()):
    for i,j in ti.ndrange(n, n):
		count = 0
        if i-1 >= 0:
			A[ind(i,j), ind(i-1,j)] += 1
			count += 1
		if i+1 < n:
			A[ind(i,j), ind(i+1,j)] += 1
			count += 1
		if j-1 >= 0:
			A[ind(i,j), ind(i,j-1)] += 1
			count += 1
		if j+1 < n:
			A[ind(i,j), ind(i,j+1)] += 1
			count += 1
		A[ind(i,j), ind(i,j)] += -count
```

### New Features (0.8.3+)

Sparse Matrix related features to `ti.linalg`

- `ti.SparseMatrixBuilder` -> `ti.linalg.SparseMatrixBuilder`
- `ti.SparseSolver` -> `ti.linalg.SparseSolver`
- `ti.sparse_matrix_builder()` -> `ti.linalg.sparse_matrix_builder()`

## Debugging a Taichi Project

### Print Results

Taichi still **not** supports **break point** debugging

- Use **run-time print** in the taichi scope to check every part 

  - The print order will not be in sequence unless `ti.sync` is used to sync the program threads

  - Print requires a system call. It can dramatically decrease the performance

  - Comma-separated parameters in the Taichi scope not supported

    ``` python
    @ti.kernel
    def foo():
    	print('a[0] = ', a[0])		 # right
    	print(f'a[0] = {a[0]}')		 # wrong, f-string is not supported
    	print("a[0] = %f" % a[0])	 # wrong, formatted string is not supported
    ```

- **Compile-time Print**: `ti.static_print` less performance loss (only at compile time)

  Print python-scope objects and constants in Taichi scope  

  ``` python
  x = ti.field(ti.f32, (2, 3))
  y = 1
  A = ti.Matrix([[1, 2], [3, 4], [5, 6]])
  
  @ti.kernel
  def inside_taichi_scope():
      ti.static_print(y)
      # => 1
      ti.static_print(x.shape)
      # => (2, 3)
      ti.static_print(A.n)
      # => 3
      for i in range(4):
          ti.static_print(A.m)
          # => 2
          # will only print once
  ```

### Visualize Fields

Print a Taichi field / numpy array will truncate your results

``` python
x = ti.field(ti.f32, (256, 256))

@ti.kernel
def foo():
	for i,j in x:
		x[i,j] = (i+j)/512.0

foo()	
# print(x)	# ok to print but not complete version (neglected mid elements)
print(x.to_numpy().tolist())	# turn the field into a list and print the full version
```

This method is hard to visualize -> use GUI / GGUI (normal dir / vel. dir / physics / …)

``` python
foo()

gui = ti.GUI("Debug", (256, 256))
while gui.running:
	gui.set_image(x)
	gui.show()
```

### Debug Mode

The debug mode is **off** by **default**

``` python
ti.init(arch=ti.cpu, debug=True)
```

- for example: if `x = ti.field(ti.f32, shape = 4)` and the field has not been def. and use `print(x[4])` 

  - Normal result: `x[4] = 0.0`
  - Debug mode: `RuntimeError` (actually not defined yet. Not safe to use)

- **Run-time Assertion**

  When debug mode is on, an **assertion** failure triggers a RuntimeError

  ``` python
  @ti.kernel
  def do_sqrt_all():
      for i in x:
          assert x[i] >= 0	# test if larger than 0, if not show RuntimeError
          x[i] = ti.sqrt(x[i])
  ```

  But the assertion will be ignored in release mode. Real computing should not be in the assertion.

  - **Traceback**: `line xx, in <module> func0()` => cannot trace to with function but only to the **kernel** (not good)

- **Compile-time Assertion**: `ti.static_assert(cond, msg=None)`

  - No run-time costs; 
  - Useful on **data types** / **dimensionality** / **shapes**; 
  - Works on **release** mode

  ``` python
  @ti.func
  def copy(dst: ti.template(), src: ti.template()):
  	ti.static_assert(dst.shape == src.shape, "copy() needs src and dst fields to be same shape")
  	for I in ti.grouped(src):
  		dst[I] = src[I]
  	return x % 2 == 1
  ```

  - **Traceback**: can trace which function is actually error

    => `excepthook=True` (at initilization): Pretty mode (stack traceback)

- Turn of Optimizations from Taichi

  - Turn off **parallelization**:

    ``` python
    ti.init(arch=ti.cpu, cpu_max_num_threads=1)
    ```

  - Turn off **advanced optimization**

    ``` python
    ti.init(advanced_optimization=False)
    ```

- Keep the Code Executable

  Check the code everytime when finish a **building block**. Keep the entire codebase **executable**.

### Other Problems

- **Data Race**

  - The for loop of the outermost scope is parallelized automatically, use `x += ()` other than normal `x = x + ()` to apply atomic protection
  - In some neighbor update scenarioes, use another field other than just one field: `y[i] += x[i-1]` because don’t know whether the neighbor had been updated or not

- **Calling Python functions from Taichi scope**

  This triggers an **undefined behaviour**

  Including functions from other imported python packages (numpy, for example) also cause this problem

  ``` python
  def foo():
      ...
      ...
      
  @ti.kernel
  def bar():
      foo()	# not from the Taichi scope => problem
  ```

- **Copying 2 var in the Python Scope**

  `b = a`: for fields this behaviour could result in pointer points to `a` other than real copy (if change b, a will also be changed)

  => `b.copy_from(a)`

- The **Data Types in Taichi should be Static**

  for example, in a loop, def. `sum = 0` at first makes the `sum` an **integer** => precision loss

- The **Data in Taichi Have a Static Lexical Scope**

  ``` python
  @ti.kernel
  def foo():
      i = 20	# this is not useful even in Python scope (not recommended)
      
      for i in range(10):	# Error
          ...
      print (i)
      
  foo()
  ```

- **Multiple Returns in a Single `@ti.func` not Supported**

  ``` python
  @ti.func
  def abs(x):
      if x >= 0:		# res = x
          return x	# if x < 0: res = -x (Correct) => return res
      else:
          return -x	# NOT SUPPORTED
  ```

- **Data Access Using Slices not Supported**

  ``` python
  M = ti.Matrix(4, 4)
  ...
  
  M_sub = M[1:2, 1:2]		# NOT SUPPORTED
  M_sub = M[(1,2), (1,2)]	# NOT SUPPORTED
  ```

## Optimizing Taichi Code

### Performance Profiling

-> Amdahl’s Law (for higher run-time cost optimization will be more valueable)

**Profiler**

- Enable Taichi’s kernel profiler:	`ti.init(kernel_profiler=True, arch=ti.gpu)`
- Output the profiling info:	`ti.print_kernel_profile_info('count')` (At last)
- Clear profiling info:	`ti.clear_kernel_profile_info()` (Useful for neglecting the runtime costs before the real-wanted loop)

**Demo**

``` 
=========================================================================
Kernel Profiler(count) @ CUDA
=========================================================================
[ % total count | min avg max ] Kernel name
-------------------------------------------------------------------------
[ 82.20% 0.002 s 17x | 0.090 0.105 0.113 ms] computeForce_c38_0_kernel_8_range_for			 [<- COSTS MOST]
[ 3.90% 0.000 s 17x | 0.002 0.005 0.013 ms] matrix_to_ext_arr_c14_1_kernel_12_range_for
[ 3.17% 0.000 s 17x | 0.003 0.004 0.004 ms] computeForce_c34_0_kernel_5_range_for
[ 2.17% 0.000 s 17x | 0.002 0.003 0.004 ms] matrix_to_ext_arr_c14_0_kernel_11_range_for
[ 2.16% 0.000 s 17x | 0.002 0.003 0.004 ms] update_c36_0_kernel_9_range_for
[ 2.09% 0.000 s 17x | 0.002 0.003 0.004 ms] computeForce_c34_0_kernel_6_range_for
[ 2.05% 0.000 s 17x | 0.002 0.003 0.003 ms] update_c36_1_kernel_10_range_for
[ 1.98% 0.000 s 17x | 0.002 0.003 0.003 ms] computeForce_c38_0_kernel_7_range_for
[ 0.28% 0.000 s 2x | 0.002 0.003 0.004 ms] jit_evaluator_2_kernel_4_serial
-------------------------------------------------------------------------
[100.00%] Total execution time: 0.002 s number of results: 9
=========================================================================
```

### Performance Tuning Tips

#### BG: Thread hierarchy of Taichi in GPU

- Iteration (Orange): each iteration in a for-loop
- Thread (Blue): the minimal parallelizable unit
- Block (Green): threads are grouped in blocks with shared block local storage (Yellow)
- Grid (Grey): the minimal unit that being launched from the host 

![image-20211022211251885](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211022211251885.png)

#### The **Block Local Storage (BLS)**

- Implemented using the shared memory in GPU
- Fast to read/write but small in size

![image-20211022211606701](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211022211606701.png)

#### **Block Size** of an hierarchically defined field (SNode-tree):

``` python
a = ti.field(ti.f32)
# 'a' has a block size of 4x4
ti.root.pointer(ti.ij, 32).dense(ti.ij, 4).place(a)
```

bls_size = 4 x 4 (x 4 Bytes) = 64 Bytes (Fast)

![image-20211022211820891](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211022211820891.png)

#### **Decide the size of the blocks**

`ti.block_dim()` before a parallel for-looop

(default_block_dim = 256 Bytes)

``` python
@ti.kernel
def func():
	for i in range(8192): # no decorator, use default settings
		...
        
	ti.block_dim(128) # change the property of next for-loop:
	for i in range(8192): # will be parallelized with block_dim=256
		...
        
	for i in range(8192): # no decorator, use default settings
        ...
```

#### **Cache Most Freq-Used Data in BLS Manually**

`ti.block_local()`

when a data is very important (actually in our case some neighbor will be counted as well)

``` python
a = ti.field(ti.f32)
# `a` has a block size of 4x4
ti.root.pointer(ti.ij, 32).dense(ti.ij, 4).place(a)

@ti.kernel
def foo():
	# Taichi will cache `a` into the CUDA shared memory
	ti.block_local(a)
	for i, j in a:
		print(a[i - 1, j], a[i, j + 2])
```

bls_size = 5 x 6 (x 4 Bytes) = 120 Bytes (the overlapped part will be cached in different bls)

![image-20211022212537538](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211022212537538.png)



# Lecture 5  Procedural Animation (21.10.26)

### Procedure Animation in Taichi 

#### Steps

1. Setup the canvas
2. Put colors on the canvas
3. Draw a basic unit
4. Repeat the basic units (tiles / fractals)
5. Animate the pictures
6. Introduce some randomness (chaos)

#### Demo: Basic Canvas Creation

``` python
import taichi as ti
ti.init(arch = ti.cuda)

res_x = 512
res_y = 512
pixels = ti.Vector.field(3, ti.f32, shape=(res_x, res_y))

@ti.kernel
def render():
    # draw sth on the canvas
    for i, j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])	# init the canvas to black
        pixels[i, j] = color
        
gui = ti.GUI("Canvas", res=(res_x, res_y))

for i in range(100000):
    render()
    gui.set_image(pixels)
    gui.show()
```

#### Colors

Add color via a for-loop

``` python
@ti.kernel
def render(t:ti.f32):
    for i,j in pixels:
    	r = 0.5 * ti.sin(float(i) / res_x) + 0.5
        g = 0.5 * ti.sin(float(i) / res_y + 2) + 0.5
        b = 0.5 * ti.sin(float(i) / res_x + 4) + 0.5
        color = ti.Vector([r, g, b])
        pixels[i, j] = color
```

#### Basic Unit

Draw a circle

``` python
@ti.kernel
def render(t:ti.f32):
    for i,j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0])	# init to black
        pos = ti.Vector([i, j])
        center = ti.Vector([res_x/2.0, res_y/2.0])
        r1 = 100.0
        r = (pos - center).norm()
        
        if r < r1:
            color = ti.Vector([1.0, 1.0, 1.0])
            
        pixels[i, j] = color
```

#### Helper Functions

- **Step**

  ``` python
  @ti.func
  def step(edge, v):
      ret = 0.0
      if (v < edge): ret = 0.0
      else: ret = 1.0
  	return ret            
  ```

- **Linearstep** (ramp)

  ``` python
  @ti.func
  def linearstep(edge1, edge2, v):
      assert(edge1 != edge2)
      t = (v - edge1) / float(edge2 - edge1)
      t = clamp(t, 0.0, 1.0)
      
      return t	# can also do other interpolation methods 
  	# such as: return (3-2*t) * t**2 (plot below)
  ```

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211029010242075.png" alt="image-20211029010242075" style="zoom:50%;" />

#### Demo: Basic Unit with Blur

``` python
@ti.func
def circle(pos, center, radius, blur):
    r = (pos - center).norm()
    t = 0.0
    if blur > 1.0: blur = 1.0
    if blur <= 0.0:
        t = 1.0-hsf.step(1.0, r/radius)
    else:
        t = hsf.smoothstep(1.0, 1.0-blur, r/radius)
    return t

@ti.kernel
def render(t:ti.f32):
    for i,j in pixels:
        ...
        
	c = circle(pos, center, r1, 0.1)
	color = ti.Vector([1.0, 1.0, 1.0]) * c
	pixels[i, j] = color
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102155632535.png" alt="image-20211102155632535" style="zoom:67%;" />

#### Repeat the Basic Units: Tiles  

``` python
@ti.kernel
def render(t:ti.f32):
    # draw something on your canvas
    for i,j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0]) # init your canvas to black
        
        tile_size = 64        
        
        center = ti.Vector([tile_size//2, tile_size//2])
        radius = tile_size//2
        
        pos = ti.Vector([hsf.mod(i, tile_size), hsf.mod(j, tile_size)])	# scale i, j to [0, tile_size-1]
        
        c = circle(pos, center, radius, 0.1)
        
        color += ti.Vector([1.0, 1.0, 1.0])*c
        
        pixels[i,j] = color
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102155844798.png" alt="image-20211102155844798" style="zoom:67%;" />

#### Repeat the Basic Units: Fractals

``` python
@ti.kernel
def render(t:ti.f32):
    # draw something on your canvas
    for i,j in pixels:
        color = ti.Vector([0.0, 0.0, 0.0]) # init your canvas to black
        tile_size = 16
        for k in range(3):
            center = ti.Vector([tile_size//2, tile_size//2])
            radius = tile_size//2
            
            pos = ti.Vector([hsf.mod(i, tile_size), hsf.mod(j, tile_size)]) # scale i, j to [0, tile_size-1]
            c = circle(pos, center, radius, 0.1)
            
            color += ti.Vector([1.0, 1.0, 1.0])*c
            
            color /= 2
            tile_size *= 2
            
        pixels[i,j] = color
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102160145427.png" alt="image-20211102160145427" style="zoom:67%;" />

#### Animate the Picture

``` python
@ti.kernel
def render(t:ti.f32):	# this t represents time (or other par) and added in the followed expressions
    # draw something on your canvas
    for i,j in pixels:
        r = 0.5 * ti.sin(t+float(i) / res_x) + 0.5
        g = 0.5 * ti.sin(t+float(j) / res_y + 2) + 0.5
        b = 0.5 * ti.sin(t+float(i) / res_x + 4) + 0.5
        color = ti.Vector([r, g, b])
        pixels[i, j] = color
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102160717368.png" alt="image-20211102160717368" style="zoom:67%;" />

#### Introduce randomness (chaos) 

- `y = rand(x)` / `y = ti.random()` (white noise, need to find a balance in smooth and chaos) 

- make in [0, 1]: `y = fract(sin(x) * 1.0)`

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102160727679.png" alt="image-20211102160727679" style="zoom: 67%;" />

- scale up: `y = fract(sin(x) * 10000.0)`

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211102160816807.png" alt="image-20211102160816807" style="zoom:67%;" />

- ``` python
  blur = hsf.fract(ti.sin(float(0.1 * t + i // tile_size * 5 + j // tile_size * 3)))
  c = circle (pos, center, radius, blur)                 
  ```

The Balance: **Perlin noise **(huge randomness but continuous in smaller scales)

-> shadertoy.com



# Lecture 6  Ray Tracing (21.11.2)

## Basis of Ray Tracing

### Rendering Types

- Realtime Rendering: Rasterization
- Offline Rendering: Ray Tracing

### Assumptions of Light Rays

- Light rays
  - go in straight lines
  - do not collide with each other
  - are reversible

## Applying Ray Tracing (Color)

In color finding, option 1-2 usually use rasterization than RT. The classical RT is option 3 and the modern RT is actually option 4 (path tracing)

### Option 1: The Color of the Object

For 256x128:

- **Ray-tracing** style:
  - Generate 256x128 rays in 3D
  - Check Ray-triangle intersection 256x128 times in 3D  
- **Rasterization** style
  - Project 3 points into the 2D plane
  - Check if a point is inside the triangle 256x128 times in 2D  

![image-20211104110341752](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104110341752.png) 

#### Flat-looking Results

Cannot tell the materials using their colors  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104110631893.png" alt="image-20211104110631893" style="zoom:50%;" /> 

**What we see = color * brightness**

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104110836230.png" alt="image-20211104110836230" style="zoom:67%;" /> 

### Option 2: Color + Shading

#### Lambertian Reflectance Model  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104111217754.png" alt="image-20211104111217754" style="zoom: 50%;" /> 

**Brightness = $\cos \theta$** 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104111341624.png" alt="image-20211104111341624" style="zoom: 50%;" /> 

The larger $\cos \theta$, the higher energy/area

#### Results with Lambertian

Looks like 3D, but still lack of the specular surfaces 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104111638925.png" alt="image-20211104111638925" style="zoom:50%;" /> 

#### Phong Reflectance Model

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104111858045.png" alt="image-20211104111858045" style="zoom:50%;" /> 

**Brightness = $(V\cdot R)^{\alpha} = (\cos(\theta))^{\alpha}$**; $\alpha$ is the hardness (intensity)

For higher $\alpha$ the visible angle will be smaller (Left: $\alpha = 2$; Right: $\alpha = 5$, more like a pulse)

![image-20211104112114526](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104112114526.png) ![image-20211104112126747](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104112126747.png)

#### Blinn-Phong Reflectance Model

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104112508191.png" alt="image-20211104112508191" style="zoom:50%;" /> 

**Brightness = $(N\cdot H)^{\alpha} = (\cos(\theta))^{\alpha^{\prime}}$**, $H = \frac{V+L}{||V+L||}$, $\alpha^{\prime}>\alpha$ is the hardness in Blinn-Phong

#### Blinn-Phong Shading Model

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104112601169.png" alt="image-20211104112601169" style="zoom:70%;" /> 

#### Results with Blinn-Phong

Shining but floating, no glassy like

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104112711501.png" alt="image-20211104112711501" style="zoom:50%;" /> 

### Option 3: The Whitted-Style Ray Tracer

-> Shadow / Mirror / Dielectric 

#### The Whitted-Style

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104115105152.png" alt="image-20211104115105152" style="zoom:50%;" /> 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG/img/image-20211104150903591.png" alt="Img" style="zoom:50%;" /> 

Since we have partially tranparent objects (the one with a gray shadow)

**Refraction** + **Reflection** (Recursive)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104151508399.png" alt="image-20211104151508399" style="zoom:50%;" /> 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104151822500.png" alt="image-20211104151822500" style="zoom:50%;" /> 

-> not easy to program 

#### Problems

- Ceiling not lightened by the big light
- The huge light source only creates hard shadows (should be softer)
- No specular spots of the light on the ground through the transparent objects

### Option 4: Path Tracer (Modern)

#### From Previous Approaches

- Shading Models:
  - The brightness matters
- The Whitted-Style Ray Tracing:
  - Getting color recursively: what color does this ray see?

#### Global Illumination (GI)

With GI, **diffuse surface** will still scatter rays as well. Without GI, if not shined from the light source directly, it has fully dark shadow

An “unified” model for different surfaces:

| Diffuse (Monte Carlo Method)                                 | Specular                                                     | Dielectric                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20211104153150409](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104153150409.png) | ![image-20211104153218601](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104153218601.png) | ![image-20211104153236293](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104153236293.png) |
| $L_o = \frac{1}{N} \sum^{N}_{k=1} L_{i,k}\ \cos(\theta_k)$   | $L_o = L_i$                                                  | $L_o = \frac{1}{N} \sum^{N}_{k=1} L_{i,k}$                   |

Monte Carlo Methdo: rely on **repeated random sampling** to obtain numerical results

All the collected **weighted average color** show the final color of a point

#### Problem in Sampling

- ##### **Expensive** (if $N \neq 1$ exponential cost too heavily)

- **Noisy** (if $N = 1$, too low sample points)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104160323829.png" alt="image-20211104160323829" style="zoom:33%;" /> 

- **Solution**: $N = 1$ but add SPP (Samples per pixel, sampling rate), which is the average of (for example, 100) times of sampling in every pixel, every time $N = 1$ 

#### Problems in Stop Criterion

The current **stop criterion**

- Hit a **light source**
  - Returns the color of the light source (usually [1.0, 1.0, 1.0])
- Hit the **bg** (casted to the void)
  - Returns the bg color (usually [0.0, 0.0, 0.0])

-> **Problem**: **Infinity loops**

- **Solution 1**: Set **depth of recursion** (stop at the n^th^ recursion): affects the **illumination**

  ![image-20211104160748348](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211104160748348.png)

- **Solution 2:** **Russian Roulette**

  When asked “what color does the ray see”

  - Set a probability **p_RR** (for instance 90%)
  - Roll a dice (0-1)
  - if roll() > p_RR:
    - stop recursion
    - return 0
  - else:
    - go on recursion: what is $L_i$
    - return $L_o/p\_RR$

  ``` python
  def what_color_does_this_ray_see(ray_o, ray_dir):	# original point and dir
      if (random() > p_RR):
          return 0
      else:
          flag, P, N, material = first_hit(ray_o, ray_dir, scene)
          if flag == False:	# hit void
              return 0
          if material.type == LIGHT_SOURCE:	# hit light source
              return 1
          else:	# recursive
              ray2_o = P
              ray2_dir = scatter(ray_dir, P, N)
              # the cos(theta) in DIFFUSE is hidden in the scatter function
              L_i = what_color_does_this_ray_see(ray2_o, ray2_dir)
              L_o = material.color * L_i / p_RR
              return L_o
  ```

#### Core Ideas Summary

- Diffuse surfaces scatter light rays as well: **Monte Carlo**
- Every hit results in ONE scattered ray: But we **sample** every pixel multiple times  
- Add the stop criterion: **Russian Roulette **(**Depth caps** are usually enabled too)

### Further Readings

- Radiometry

- The rendering equation
  $$
  L_o = L_e + \int_{\Omega} L_i \cdot f_r \cdot \cos \theta \cdot \mathrm{d}\omega
  $$



# Lecture 7  Ray Tracing 2 (21.11.9)

## Recap

- Color
  - RGB Channels
  - Range $\in [0.0, 1.0]$
  - As a “filter”
- Brightness: 
  - Power per unit solid angle per unit projected area $[\mathrm{\frac{lm}{sr\cdot m^2}}]$ or $[\mathrm{\frac{W}{sr\cdot m^2}}]$ 
  - Range $\in [0.0, +\infty]$
  - Called **Radiance** in Radiometry
- What we see = Color * Brightness
- What we see after multiple bounces = Color * Color * … * Brightness (Ray tracing)

## Ray-casting from Camera/Eye

### Ray

Ray is a line def by its **origin** ($O$) and **dir** ($\vb{d}$) (or a point it passed by)
$$
P = O + t\vb{d}\ ; \quad \vb{d} = \frac{P-O}{||P-O||}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110220057903.png" alt="image-20211110220057903" style="zoom:50%;" />

### Camera/Eye and Monitor

- **Positioning camera / eye**

  ``` python
  lookfrom[None] = [x, y, z]
  ```

- **Orienting camera / eye**

  ``` python
  lookat[None] = [x, y, z]
  ```

- **Placing the screen**

  ``` python
  # center pass through lookat-lookfrom
  # Perpendicular with lookat-lookfrom
  ```

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110223353753.png" alt="image-20211110223353753" style="zoom:67%;" />

  ``` python
  distance = 1.0

- **Orienting the screen** (up vector)

  ``` python
  up[None] = [0.0, 1.0, 0.0]
  ```

- **Size of the screen** (**Field of View**)

  The FOV setting to match to the corresponding FOV of the real life will be better (i.e. smaller monitor / farther from the screen -> smaller FOV)

  ``` python
  theta = 1.0/3.0 * PI
  half_height = ti.tan(theta / 2.0) * distance
  half_width = aspect_ratio * half_height * distance	# can also use another FOV to control
  ```

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110223610044.png" alt="image-20211110223610044" style="zoom:80%;" />	<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110224012066.png" alt="image-20211110224012066" style="zoom:80%;" />

  ``` python
  w = (lookfrom[None]-lookat[None]).normalized()
  u = (up[None].cross(w)).normalized()
  v = w.cross(u)
  ```

  ``` python
  cam_lower_left_corner[None] = (lookfrom[None] - half_width * u - half_height * v – w)* distance
  cam_horizontal[None] = 2 * half_width * u * distance
  cam_vertical[None] = 2 * half_height * v * distance
  ```

### Ray-casting

``` python
u = float(i)/res_x
v = float(j)/res_y # uv in [0, 1]
ray.direction = cam_lower_left_corner[None] + u * cam_horizontal[None] + v * cam_vertical[None] - lookfrom[None] 
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110224200599.png" alt="image-20211110224200599" style="zoom:67%;" />

A pixel has its size as well -> + 0.5 pixels

``` python
u = float(i+0.5)/res_x
v = float(j+0.5)/res_y # uv in (0, 1)
```

## Ray-object Intersection

### Sphere

#### Sphere Intersection

- Def of the sphere: $||P-C||^2 - r^2 = 0$ 

- Intersection? : $||O+t\vb{d}-C||^2 - r^2 = 0$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110233534895.png" alt="image-20211110233534895" style="zoom:80%;" />

  => $d^{T} d t^{2}+2 d^{T}(O-C) t+(O-C)^{\mathrm{T}}(O-C)-\mathrm{r}^{2}=0$ (In the form of $at^2 + bt +c = 0$)

  $b^2 - 4ac$: $>0$ (2 roots, intersect); $= 0$ (1 root, tangential line); $<0$ (no root, no intersect)

  Find the **smallest positive root** $t = \frac{-b \pm \sqrt{b^{2}-4 a c}}{2 a},\ t>0$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211110235745554.png" alt="image-20211110235745554" style="zoom:80%;" />

  **Problem**: Shadow Acne (Caused by precision)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111000644550.png" alt="image-20211111000644550" style="zoom:50%;" />

  -> Want a **slightly more positive** number than 0: $t = ..., t > \epsilon $ (for instance $\epsilon = 0.001$)

  For example in the following picture: $t_1$ could be the sphere itself

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111003704364.png" alt="image-20211111003704364" style="zoom: 80%;" />

#### Cornell Box

Actually formed by 4 huge spheres other than using planes (more convenient)

![image-20211111003937029](https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111003937029.png)

### Plane

#### Ray-plane Intersection

- Definition of a plane: $(P-C)^{\mathrm{T}} N=0$

- Intersection: $(O+t\vb{d}-C)^{\mathrm{T}} N =0$ (Has a root as long as $d^{\mathrm{T}}N \ne 0$)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111004240797.png" alt="image-20211111004240797" style="zoom:50%;" />

- $O+t\vb{d}$ inside triangle? -> **Barycentric Coordinate**

  $S_{\Delta PAB} = \frac{1}{2} ||\mathrm{PA\times PB}||$ 

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111005205251.png" alt="image-20211111005205251" style="zoom:80%;" />

### Ray-object Intersection

- **Implicit surfaces:**
  - Find its surface definition
  - Plug the ray equation into the surface definition
  - Look for the smallest positive $t$  
- **Polygonal surfaces:** (Polygon meshes are usually made of triangles)
  - Loop over all its polygons (usually triangles)
  - Find the ray-polygon (triangle) intersection with the smallest positive $t$

## Sampling

Want to sample the directions of rays uniformly  

### Coordinates

- Cartesian coordinates: $[x, y, z]$
- Polar coordinates: $[r, \phi, \theta]$
  - $x = r\cdot \cos\phi \cdot \sin\theta$
  - $z = r \cdot \sin\phi \cdot \cos\theta$
  - $y = r\cdot \cos \theta$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111101215134.png" alt="image-20211111101215134" style="zoom:80%;" />

### Sampling the Hemisphere Uniformly

1. Attempt one: $\phi = \text{rand} (0,2\pi),\ \theta = \text{rand}(0,\pi),\ r = 1$ => Not uniform after mapping to the polar coord

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111101937097.png" alt="image-20211111101937097" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111101951187.png" alt="image-20211111101951187" style="zoom:60%;" />

   -> **Uniform**?: The probability of sampling is proportional to the surface area (differential surface element: $\mathrm{d}A = r^2 \sin\theta\ \mathrm{d}\theta\mathrm{d}\phi $) 

   -> probability density function (p.d.f.)  $f(\theta)=\int_{0}^{2 \pi} f(\phi, \theta)\ \mathrm{d} \phi=\frac{\sin (\theta)}{2}$ 

2. The **corrected attempt**: $\phi = \text{rand} (0,2\pi),\ \theta = \arccos(\text{rand}(-1,1)),\ r = 1$ (**inverse transform sampling**) => Much more uniform

   <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111104749948.png" alt="image-20211111104749948" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111104802197.png" alt="image-20211111104802197" style="zoom:60%;" />

3. Negate the direction if against the normal

   ### Sampling a Sphere

-  $\phi = \text{rand} (0,2\pi),\ \theta=\arccos(\text{rand}(-1,1)),\ r = \sqrt[3]{\text{rand(0,1)}}$ 
-  The **rejection** method: (Higher rejection rate for higher dimension problem (**higher costs**))
   -  Sample inside a uniform sphere: $x = \text{rand(-1,1)}, y = \text{rand(-1,1)}, z = \text{rand(-1,1)}$; Reject if $x^2 + y^2 + z^2 > 1$ and resample
   -  Sample on a uniform sphere: Sample inside a uniform sphere and project

  ### Importance Sampling

- A $\cos(\theta)$ weighted importance sampling: $\theta=\arccos(\sqrt{\text{rand}(-1,1)})$

- Alternative: uniformly sample a point on a uniform sphere centered at $P+N$, says $ S$; then normalize $S-P$ as the cos-weighted sampled dir

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111112117199.png" alt="image-20211111112117199" style="zoom:80%;" />  normalized<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111111818285.png" alt="image-20211111111818285" style="zoom:80%;" />  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111111939262.png" alt="image-20211111111939262" style="zoom: 50%;" />

## Reflection v.s. Refraction

- **Law of reflection**: $\theta_i = \theta_r$

- **Snell’s law** (for refraction): $n_1\sin(\theta_i) = n_2\sin(\theta_t)$

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111112536089.png" alt="image-20211111112536089" style="zoom:80%;" />

### **Total Reflection**

Happens when $n_1> n_2$ (e.g., from glass to air)

Snell’s law may fail to give $\sin(\theta_t) = \frac{n_1}{n _2} \sin(\theta_i)>1$ => fail to solve $\arcsin(\theta_t)$ （only reflection)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111112839174.png" alt="image-20211111112839174" style="zoom:80%;" />

### Reflection Coefficient R

At a steep angle => **Reflection Coefficient $R$** (how much been reflected, material ($n_1$ & $n_2$) and view point ($\theta$) dep)

**Refraction Coefficient** $T = 1-R$ 

#### Fresnel’s Equation

- S-polarization (perpendicular to)
  $$
  R_{S}=\left(\frac{n_{1} \cos \left(\theta_{i}\right)-n_{2} \cos \left(\theta_{t}\right)}{n_{1} \cos \left(\theta_{i}\right)+n_{2} \cos \left(\theta_{t}\right)}\right)^{2}
  $$

- P-polarization (parallel to)
  $$
  R_{P}=\left(\frac{n_{1} \cos \left(\theta_{t}\right)-n_{2} \cos \left(\theta_{i}\right)}{n_{1} \cos \left(\theta_{t}\right)+n_{2} \cos \left(\theta_{i}\right)}\right)^{2}
  $$

- For “natural light”: $R = \frac{1}{2} (R_S + R_P)$ 

#### Schlick’s Approximation

Material and angle
$$
R(\theta_i) = R_0 + (1-R_0)(1-\cos(\theta_i))^5\ ;\quad R_0 = \left(\frac{n_1-n_2}{n_1+n_2} \right)^2
$$

### Path Tracing with R

``` python
def scatter_on_a_dielectric_surface(I):
	sin_theta_i = -I.cross(N)
    theta_i = arcsin(sin_theta_i)
    if n1/n2*sin_theta_i > 1.0:
        return R # total internal reflection 
    else:
        R_c = reflectance(theta_i, n1, n2)
        if random() <= R_c:
            return R # reflection (in = out)
        else:
            return T # refraction (Snell's law)
```

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211111133711072.png" alt="image-20211111133711072" style="zoom:80%;" />

## Recursion in Taichi

Call functions: temp var. -> storage in stack => higher pressure

Better solution -> using loops for tail-resursion

Optimization: Remain the fronter part of breaking loops; modify the tail-recursion part:

``` python
...
	...	
    	...
        	else: 
            	brightness *= material.color / p_RR
            	ray_o = P
            	ray_dir = scatter(ray_dir, P, N)
            	# the cos(theta) in DIFFUSE is hidden in the scatter function 
	return color           
```

## Anti-Aliasing

Zig-zag artifacts -> softening the edges

In this case we can use 4 times of random sampling -> anti-aliasing



# Lecture 8  Deformable Simulation 01: Spatial and Temporal Discretization (21.11.16)

  ## Laws of Physics

### Equation of Motion

- Define: $\frac{\mathrm{d}}{\mathrm{d}t} q \equiv \dot q$
  - $\dot x = v$ ;  $\dot v = a$ ; $\ddot {x} = a$   
  
- Linear ODE: $M\ddot{x} = f(x)$

  - For linear materials, $f(x) = -K(x-X)$ => $M\ddot{x} + K(x-X) = 0$ or $M\ddot{u} + Ku = 0$ (be def $u\equiv x-X$) 

    Widely used for small deformation, such as physically based **sound simulation** (rigid bodies) and **topology optimization**

  - General Cases: $\dot x = v$ ; $\dot v = a = M^{-1}f$ 


## Integration in Time

**Equation of Motion**

Use $h$ as the step size

- $x\left(t_{n}+h\right)=x\left(t_{n}\right)+\int_{0}^{h} v\left(t_{n}+t\right)\ \mathrm  d t$
- $v\left(t_{n}+h\right)=v\left(t_{n}\right)+\int_{0}^{h} M^{-1}f\left(t_{n}+t\right) \ \mathrm d t$

### Explicit (forward) Euler Integration

- $x_{n+1} = x_n + hv_n$ 
- $v_{n+1} = v_n + hM^{-1}f(x_n)$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118154017210.png" alt="image-20211118154017210" style="zoom:50%;" />

**Extremely fast**, but **increase the system enery** gradually (leads to explode) -> **seldom used** => **Symplectic Euler Integration**

### Symplectic Euler Integration

- $v_{n+1} = v_n + hM^{-1}f(x_n)$ 
- $x_{n+1} = x_n + hv_{n+1}$ 

Also very fast. Update the velocity first -> momentum preserving, **oscillating system Hamiltonian** (could be unstable) -> Widely used in **accuracy centric applications** (astronomy simulation / molecular dynamics /…)

### Implicit (backward) Euler Integration

- $v_{n+1} = v_n + hM^{-1}f(x_{n+1})$
- $x_{n+1} = x_n + hv_{n+1}$

Often **expensive**. Energy declines, **damping the Hamitonian** from the osscillating components. Often **stable for large timesteps** -> Widely used in **performance-centric applications** (game / MR / design / animation)

## Integration in Space

### Mass-Spring System

- Tessellate the mesh into a **discrete** one
- Aggregate the **volume mass** to **vertices**
- Link the mass-vertices with **springs**

#### Deformation

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118164920620.png" alt="image-20211118164920620" style="zoom:50%;" />

- Spring current pose: $x_1$, $x_2$
- Spring current length: $l = ||x_1 - x_2||$
- Spring rest-length: $l_0$
- Deformation: $l-l_0$ 
- Deformation (Elastic) evergy (Huke’s Law): $E\left(x_{1}, x_{2}\right)=\frac{1}{2} k\left(l-l_{0}\right)^{2}$
- Gradient:
  - $\frac{\partial E}{\partial x_{1}}=\frac{\partial l}{\partial x_{1}} \cdot \frac{\partial E}{\partial l}=\frac{x_{1}-x_{2}}{l_{0}} \cdot k\left(l-l_{0}\right)$
  - $f(x_1) = -\frac{\partial E}{\partial x_1}$
  - $f(x_2) = -f(x_1)$ 

#### Demo

Compute force

``` python
@ti.kernel
def compute_gradient():
# clear gradient
	for i in range(N_edges):
		grad[i] = ti.Vector([0, 0])

	# gradient of elastic potential
	for i in range(N_edges):
		a, b = edges[i][0], edges[i][1]
		r = x[a]-x[b]
        l = r.norm()
        l0 = spring_length[i]
        k = YoungsModulus[None]*l0
        # stiffness in Hooke's law
        gradient = k*(l-l0)*r/l
        grad[a] += gradient
        grad[b] += -gradient
```

Time integration

``` python
# symplectic integration
acc = -grad[i]/m - ti.Vector([0.0, g])
v[i] += dh*acc
x[i] += dh*v[i]
```

#### Applications

Cloth sim / Hair sim

Not the best choice when sim **continuum area/volume**: Area/volume gets **inverted** without any penalty  => Linear FEM

### Constitutive Models

#### Deformation Map

A continuous model to describe deformation: $x = \phi(X)$ at every point (Deformation map)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118171109581.png" alt="image-20211118171109581" style="zoom:50%;" />

- Translation: $\phi(X) = X + t$
- Rotation: $\phi(X) = RX$, $R = \begin{bmatrix}\cos \theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$
- Scaling: $\phi(X) = SX$, $S = \begin{bmatrix}2 & 0 \\ 0 & 1\end{bmatrix}$

Generally: For $X$ near $X^*$: (with 1st order Taylor)
$$
\phi(X) \approx \frac{\partial \phi}{\partial X}\left(X-X^{*}\right)+\phi\left(X^{*}\right)=\underbrace{\frac{\partial \phi}{\partial X}}_F X+\underbrace{\left(\phi\left(X^{*}\right)-\frac{\partial \phi}{\partial X} X^{*}\right)}_t
\approx FX + t
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118171844688.png" alt="image-20211118171844688" style="zoom:50%;" />

($F$ - **Deformation gradient**, $F = \left[\begin{array}{ll}
\partial x_{1} / \partial X_{1} & \partial x_{1} / \partial X_{2} \\
\partial x_{2} / \partial X_{1} & \partial x_{2} / \partial X_{1}
\end{array}\right]$)

After approx, close to a translation in very small region: $F = \frac{\partial \phi}{\partial X} = I$ in translation; $F = R$ in rotation; $F = S$ in scaling

=> A **non-rigid** deformation gradient shall end up with a **non-zero deformation energy**

#### Energy Density

$$
\Psi(x) = \Psi(\phi(X)) \approx \Psi(FX +t) \approx \Psi(FX) \approx \Psi(F)
$$

- An energy density function at $x = \phi(X)$
- Energy density function should be translational invariant: $\Psi(x) = \Psi(x+t)$ 
- $X$ is the state-independent rest-pose (for elastic materials)  
- We have $\Psi = \Psi(F)$ being a function of the **local deformation gradient** alone  

=> Deformation gradient is NOT the best quantity to describe deformation  

#### Strain Tensor

Strain tensor: $\epsilon(F)$

- Descriptor of **severity of deformation**
- $\epsilon(I) = 0$
- $\epsilon(F) = \epsilon(RF)$ for $\forall R\in SO (\text{dim})$ 

Samples in different **constitutive models**:

- St. Venant-Kirchhoff model:  $\epsilon(F)=\frac{1}{2}\left(F^{T} F-I\right)$
- Co-rotated linear model:  $\epsilon(F) = S-I$, where $F =RS$

From energy density function to energy: $E (x) = \int _\Omega \Psi(F) \ \mathrm{d}X$ => Spatial Discretization  

### Linear Finite Element Method (FEM)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118175112445.png" alt="image-20211118175112445" style="zoom:50%;" />
$$
E = \int (\text{The whole area}) = \sum (\text{A piece of triangle})
$$

#### Linear FEM Energy

- Continuous Space: $E(x) = \int_{\Omega} \Psi(F(x))\ \mathrm{d}X$
- Discretized Space:
  - $E(x)=\sum_{e_{i}} \int_{\Omega_{\mathrm{e}_{\mathrm{i}}}} \Psi\left(F_{i}(x)\right)\ \mathrm d X=\sum_{e_{i}} w_{i} \Psi\left(F_{i}(x)\right)$ ($F_i = \text{const}$, indep on the pos) (Energy density x size = Energy)
  - $w_i = \int_{\Omega_{e_i}} \mathrm dX$ : size (area / volume) of the i-th elem

Find **deformation gradient**: Original pose (Uppercase) => Current pose $x_i = FX_i +t$ (share the same $F$ and $T$)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118191850161.png" alt="image-20211118191850161" style="zoom:50%;" />
$$
\underbrace{\begin{bmatrix}x_1 - x_4 & x_2-x_4 & x_3-x_4\end{bmatrix} }_{D_s}
= F \underbrace{ \begin{bmatrix}X_1 - X_4 & X_2-X_4 & X_3-X_4\end{bmatrix} }_{D_m}\ ;\quad
F = D_sD_m^{-1}
$$
Find the **gradient of $\Psi(F(x))$**

Chain rule: (Note: $B: A=A: B=\sum_{i, j} A_{i j} B_{i j}=\sqrt{\operatorname{tr}\left(A^{\mathrm T} B\right)}$)
$$
\frac{\partial \Psi}{\partial x}=\frac{\partial F}{\partial x}: \frac{\partial \Psi}{\partial F}
$$
**1st Piola-Kirchhoff stress tensor**  for hyperelastic matrial: $P = \frac{\partial \Psi}{\partial F}$

Some 1st Piola-Kirchhoff stress tensors:

- St. Venant-Kirchhoff model (StVK):  
  - Strain: $\epsilon _{\mathrm{stvk}} (F) = \frac{1}{2} (F^{\mathrm{T}}F-I)$
  - Energy density: $\Psi({F})=\mu\left\|\frac{1}{2}\left(F^{T} F-I\right)\right\|_{F}^{2}+\frac{\lambda}{2} \operatorname{tr}\left(\frac{1}{2}\left(F^{T} F-I\right)\right)^{2}$
  - $P=\frac{\partial \Psi}{\partial F}=F\left[2 \mu \epsilon_{s t v k}+\lambda t r\left(\epsilon_{s t v k}\right) I\right]$
- Co-rotated Linear Model:
  - Strain: $\epsilon _{c}(F) = S-  I$, where $F=RS$
  - Energy density: $\Psi(F)=\mu\left\|R^{T} F-I\right\|_{F}^{2}+\frac{\lambda}{2} \operatorname{tr}\left(R^{T} F-I\right)^{2}$
  - $P=\frac{\partial \Psi}{\partial F}=R\left[2 \mu \epsilon_{c}+\lambda t r\left(\epsilon_{c}\right) I\right]=2 \mu(F-R)+\lambda tr\left(R^{T} F-I\right) R$ 

#### Linear FEM

- Elastic energy: $E_i (x) = w_i\Psi (F_i (x))  $
- Gradient: $\frac{\partial E_{i}}{\partial x}=w_{i} \frac{\partial F_{i}}{\partial x}: P_{i}$
- Gradient of energy density: $\frac{\partial \Psi}{\partial x_{j}^{(k)}}=\frac{\partial F}{\partial x_{j}^{(k)}}: P$

For $\frac{\partial \Psi}{\partial x_{j}^{(k)}}=\frac{\partial F}{\partial x_{j}^{(k)}}: P$ 

-> Taichi: Autodiff

#### Demo

- General Method:

  Compute Energy

  ```python
  # gradient of elastic potential
  for i in range(N_triangles):
  	Ds = compute_D(i)
  	F = Ds@elements_Dm_inv[i]
  	# co-rotated linear elasticity
  	R = compute_R_2D(F)
  	Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
      # first Piola-Kirchhoff tensor
      P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
      #assemble to gradient
      H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
      a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
      gb = ti.Vector([H[0,0], H[1, 0]])
      gc = ti.Vector([H[0,1], H[1, 1]])
      ga = -gb-gc
      grad[a] += ga
      grad[b] += gb
      grad[c] += gc
  ```

  Time integration

  ``` python
  # symplectic integration
  acc = -grad[i]/m - ti.Vector([0.0, g])
  v[i] += dh*acc
  x[i] += dh*v[i]
  ```

- With **autodiff**

  Compute energy

  ``` python
  @ti.kernel
  def compute_total_energy():
      for i in range(N_triangles):
          Ds = compute_D(i)
          F = Ds @ elements_Dm_inv[i]
          # co-rotated linear elasticity
          R = compute_R_2D(F)
          Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
          element_energy_density = LameMu[None]*((FR)@(F-R).transpose()).trace() + 0.5*LameLa[None]*(R.transpose()@F-Eye).trace()**2
          
          total_energy[None] += element_energy_density * elements_V0[i]
  ```

  Compute gradient

  ``` python
  if using_auto_diff:
  	total_energy[None]=0
  	with ti.Tape(total_energy):
  	compute_total_energy()
  else:
  	compute_gradient()	# use this func
  ```

  Time integration

  ``` python
  # symplectic integration
  acc = -x.grad[i]/m - ti.Vector([0.0, g])	# this `x.grad` will be 2n*1 vector (same with compute_grad)
  v[i] += dh*acc
  x[i] += dh*v[i]
  ```

#### Revisit Mass-spring System

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211118230855291.png" style="zoom:50%;" />

- Deformation gradient: $F=D_{S} D_{m}^{-1}=\frac{x_{1}-x_{2}}{X_{1}-X_{2}}=\frac{x_{1}-x_{2}}{l_{0}}$ (3*1 matrix)
- Deformation strain: $\epsilon = ||F||-1$
- Energy density: $\Psi=\mu \epsilon^{2}=\mu\left(\left\|\frac{x_{1}-x_{2}}{l_{0}}\right\|-1\right)^2$
- Energy: $\mathrm{E}=l_{0} \Psi=\frac{1}{2} \frac{2 \mu}{l_{0}} l_{0}^{2} \epsilon^{2}=\frac{1}{2} k\left(\left\|x_{1}-x_{2}\right\|-l_{0}\right)^{2}$  ($k$ is the Young’s Modulus / $l$)



# Lecture 9  Deformable Simulation 02: The Implicit Integration Methods  (21.11.23)

## The Implicit Euler Integration

 ### Time Step

- In simulation: The time difference between the adjacent ticks on the temporal axis for the simulation: $h_{\text{sim}}$
  - `v[i] += h * acc`
  - `x[i] += h * v[i]`
- In display: The time difference between two images displayed on the screen: $h_{\text{disp}} = 1/60\ \mathrm{s}$ for 60 Hz applications

**Sub-(time)-stepping: $n_{\text{sub}}$**: $h_{\text{sim}} = \frac{h_{\text{disp}}} {n_{\text{sub}}}$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124213141086.png" alt="image-20211124213141086" style="zoom:50%;" />

The smaller $n_{\text{sub}}$, the larger $h_{\text{sim}}$ (less accurate)

- For explicit integrations: too long time step may lead to explosion (as $h$ increases, diverge);
- For implicit: usually converge for all $h$ => artifact: slower converge

## Numerical Recipes for Implicit Integrations

### The Baraff and Witkin Style Solution

One iter of Newton’s Method, referred as semi-implicit

- $x_{n+1} = x_n + hv_{n+1}$ & $v_{n+1} = v_n + hM^{-1} f(x_{n+1})$ then $\Rightarrow x_{n+1} = x_n + hv_n + h^2 M^{-1}f(x_{n+1})$

- **Goal**: Solving $x_{n+1} = x_n + hv_n + h^2 M^{-1}f(x_{n+1})$

- **Assumption**: $x_{n+1}$ is not too far away from $x_n$

- **Algorithm**: Non-linear -> Linear ($\delta x$ - the deviation between the two timesteps, which is very small)

  - Let $\delta x = x_{n+1} - x_n$, then $f(x_{n+1}) \approx f(x_n) + \grad_{x}f(x_n)\delta x$  (1st order Taylor, neglect all the minor errors)

  - Substitute this approx:

    - $x_{n}+\delta x=x_{n}+h v_{n}+h^{2} M^{-1}\left(f\left(x_{n}\right)+\nabla_{x} f\left(x_{n}\right) \delta x\right)$
    - $\left(M-h^{2} \nabla_{x} f\left(x_{n}\right)\right) \delta x=h M v_{n}+h^{2} f\left(x_{n}\right)$  (Matrix op: (2n x 2n) x (2n x 1) = (2n x 1), sim to Ax = b)
    - $x_{n+1} = x_n + \delta x$, $v_{n+1} = \delta x/h$

    The solution is actually the location of $x_{n+1}$ while in implicit method the sol will be much more closed to the exact root (error is the red line)

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211124231048617.png" alt="image-20211124231048617" style="zoom:50%;" />

### Reformulating the Implicit Euler Problem

To reduce the red part: add another step: 

- Integrating the nonlinear root finding problem over $x$ gives:
  $$
  x_{n+1}=\operatorname{argmin}_{x}\left(\frac{1}{2}\left\|x-\left(x_{n}+h v_{n}\right)\right\|_{M}^{2}+h^{2} E(x)\right), \text { given } f(x)=\nabla_{x} E(x)
  $$
  
  Note: (Matrix Norm) $\| x\| _A = \sqrt{x^{\mathrm{T}}Ax}$; (Vector Derivative) $\grad_x (x^{\mathrm{T}}Ax) = (A +A^{\mathrm{T}})x$

#### **Minimization / Non-linear Root-finding**:

- Let $g(x) =\frac{1}{2}\left\|x-\left(x_{n}+h v_{n}\right)\right\|_{M}^{2}+h^{2} E(x)$  (Mass matrix is usually diagonalised, so the transpose is itself)

- Then  $\nabla_{x} g\left(x_{n+1}\right)=M\left(x_{n+1}-\left(x_{n}+h v_{n}\right)\right)-h^{2} f\left(x_{n+1}\right)$  (Energy of the der of pos => The fastest energy increasing dir, the negative val is force)

- For nonsingular $M$: ($g$ at an extreme => the derivatifve of $g$ at 0, the equation above equals to 0)

  $\nabla_{x} g\left(x_{n+1}\right)=0 \leftrightarrow x_{n+1}=\left(x_{n}+h v_{n}\right)+h^{2} M^{-1} f\left(x_{n+1}\right)$  (The root of implicit Euler)

#### Convex Minimization of $\min g(x)$

The general descent method:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125172532811.png" alt="image-20211125172532811" style="zoom:50%;" />

``` python
def minimize_g():
    x = x_0
    while grad_g(x).norm() > EPSILON:
        Determine a descent dir: dx
        Line search: choose a stepsize: t > 0
        Update: x = x + t * dx
```

- Determine a **descent dir**: $\mathrm{d}x $

  - Opt 1: **Gradient Descent**: $\mathrm{d}x = -\grad_x g(x)$ (use the intersect point with the x-axis)

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125172740172.png" alt="image-20211125172740172" style="zoom:50%;" />

  - Opt 2: **Newton’s Method**: $\mathrm{d} x=-\left(\nabla_{x}^{2} g(x)\right)^{-1} \nabla_{x} g(x)$ (apply the second order derivative as the curvature -> second order -> use its valley (actually also decide the step)) -> usually when a func can solve 2nd order

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125173055239.png" alt="image-20211125173055239" style="zoom:50%;" />

- Find the **step size**:
  
  - **Line search**: choose a step size $t>0$ given a $\mathrm{d}x$
  
  - **Backtracking**: 
  
    ``` python
    def line_search(): 
        t = t_0
        while g(x) + alpha*t*grad_g(x).dot(dx) < g(x+t*dx): 
            # this alpha indicates a line between the grad and the line w/ alpha = 0
            t = t * beta
            
        return t
    ```
    
    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125212312113.png" alt="image-20211125212312113" style="zoom:50%;" />
    
    $\alpha \in (0, 0.5)$, $\beta \in (0,1)$ (common choice: $\alpha = 0.03 $ & $\beta = 0.5$)

**Problem**

But most deformable bodies have non-convex energies: unsteady equilibrium.

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125213834479.png" alt="image-20211125213834479" style="zoom:50%;" />

#### Steps

$g(x)$ is contains 2 terms: dynamics and elastic
$$
g(x)=\frac{1}{2}\left\|x-\left(x_{n}+h v_{n}\right)\right\|_{M}^{2}+h^{2} E(x)
$$

1. **Init guess**: $x = x_n$ or $x = x_n + hv_n$
2. **While loop**: while not converge:
   - Descent dir: $\mathrm{d}x = -\grad_x g(x)$ or $\mathrm{d}x = -(\grad_x^2 g(x))^{-1} \grad_x g(x)$
     - Gradient: $\grad_{x} g(x)=M\left(x-\left(x_{n}+h v_{n}\right)\right)+h^{2} \grad_{x} E(x)$ (linear + non-linear) (Model dep)
     - Hessian (Matrix): $\grad_x^2 g(x= M+ h^2\grad^2_x E(x)$  (Model dep)
   - Line search: det the stepsize $t$
   - Update: $x = x + t\cdot \mathrm{d}x$ 

### Definiteness-fixed Newton’s Method

For general cases:

in Step 2 (while), After compute gradient dir and Hessian, add another substep:

- Fix Hessian to positive definite: $\tilde{H} = \mathrm{fix}(H)$

#### Definiteness-fix

$\tilde{H} = \mathrm{fix}(H)$

- Opt 1: **Global Regularization**

  - Init: $\tilde{H} = H$, `flag = False`, `reg = 1`
  - `while not flag:`
    - `flag`, `L = factorize(~H)` Try to factorize $\tilde{H} = LL^{\mathrm{T}}$ (if success -> OK; else: add identity to let it tends to indentity (definite))
    - $\tilde{H} = H + \mathrm{reg} * I$, `reg = reg * 10`

- Opt 2: **Local Regularization** (in mass-spring system)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211125232625721.png" alt="image-20211125232625721" style="zoom: 50%;" />
  $$
  \nabla_{x}^{2} E(x)=\left[\begin{array}{cccc}
  K_{1} & -K_{1} & & \\
  -K_{1} & K_{1}+K_{2}+K_{3} & -K_{3} & -K_{2} \\
  & -K_{3} & K_{3} & \\
  & -K_{2} & & K_{2}
  \end{array}\right]
  $$

  $$
  \left[\begin{array}{cc}
  K_{1} & -K_{1} \\
  -K_{1} & K_{1}
  \end{array}\right]\left[\begin{array}{cc}
  K_{2} & -K_{2} \\
  -K_{2} & K_{2}
  \end{array}\right]\left[\begin{array}{cc}
  K_{3} & -K_{3} \\
  -K_{3} & K_{3}
  \end{array}\right]
  $$

  - $\nabla_{x}^{2} g(x)=M+h^{2} \nabla_{x}^{2} E(x)=M+h^{2} \sum_{j=1}^{m} \nabla_{x}^{2} E_{j}(x)$

    - $K_{1} \geqslant 0 \Rightarrow\left[\begin{array}{cc}
      K_{1} & -K_{1} \\
      -K_{1} & K_{1}
      \end{array}\right]=K_{1} \otimes\left[\begin{array}{cc}
      1 & -1 \\
      -1 & 1
      \end{array}\right] \geqslant 0$ (a positive semi-definite matrix Otimes another .. matrix => also positive semi-definite)
    - $\Rightarrow \nabla_{x}^{2} E(x)=\nabla_{x}^{2} E_{1}(x)+\nabla_{x}^{2} E_{2}(x)+\nabla_{x}^{2} E_{3}(x) \geqslant 0$
    - $ \Rightarrow \nabla_{x}^{2} g(x)=M+h^{2} \nabla_{x}^{2} E(x)>0$ 

    Has a sufficient condition: $K_1 \ge 0,\ K_2 \ge0, \ K_3\ge0$

  - $K_{1}=k_{1}\left(I-\frac{l_{1}}{\left\|x_{1}-x_{2}\right\|}\left(I-\frac{\left(x_{1}-x_{2}\right)\left(x_{1}-x_{2}\right)^{T}}{\left\|x_{1}-x_{2}\right\|^{2}}\right)\right) \in \mathbb{R}^{2 \times 2}$

    - $K_1 = Q \Lambda Q^{T}$ (<- Eigen value decomposition)
    - $\tilde \Lambda = \max(0,\Lambda)$ (<- Clamp the negative eigen values)
    - $\tilde{K_1} = Q\tilde{\Lambda} Q^{\mathrm{T}}$ (<- Construct the p.s.d. projection)

After this definiteness-fix: (Newton’s)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211126011246122.png" alt="image-20211126011246122" style="zoom:50%;" />

## Linear Solvers  

### Linear Solvers for $Ax = b$

- **Direct solvers**

  - **Inversion**: $x = A^{-1} b$ (requires higher complexity for sparse and higher order matrices)

  - **<u>Factorization</u>** (usually better especially for sparse matrices): 

    $ A=\left\{\begin{array}{cl}
    L U & \text {, if } A \text { is a square matrix } \\
    L D L^{\mathrm{T}} & \text {, if } A=A^{T} \\
    L ^{\mathrm{T}} & \text {, if } A=A^{T} \text { and } A>0
    \end{array}\right.$

-  **Interative solvers**

  - **Stationary iterative linear solvers**: Jacobi / Gauss-Seidel / SOR / Multigrid
  - **Krylov subspace methods**: <u>Conjugate Gradient (CG)</u> / biCG / CR / MinRes / GMRes


### Factorization

For $A = L L^{\mathrm{T}}$ (Assume already p.s.d. proceeded)

- Solve $Ax = b$ is equiv to $LL^{\mathrm{T}}x = b$ 

- First solve $Ly = b$ (Forward substitution)

- Then solve $L^{\mathrm{T}} x = y$ (Backward substitution)

  For sparse matrices -> complexity not high; But not parallized (CPU backend only)

**Current APIs**:

- `[SparseMatrixBuilder] = ti.linalg.SparseMatrixBuilder()`
- `[SparseMatrix] = [SparseMatrixBuilder].build()`
- `[SparseMatrixSolver] = ti.linalg.SparseSolver(solver_type, ordering)`
- `[NumpyArray] = [SparseMastrixSolver].solve([Field])`

### Conjugate Gradient (CG) Method

**Properties**:

- Works for any symmetric positive definite matrix $A = A^{\mathrm{T}},\ A >0$ 
- Guarantees to converge in $n$ iter for $A\in\R ^{n\times n}$
- Works amazingly good if the condition number $\kappa = \lambda_{\max} / \lambda_{\min}$ (eigenvalues) of $A$ is small (when $\kappa = 1$, identity matrix)
- Short codes

**Demo**: (Python scope - control flow)

``` python
def conjugate_gradient(A, b, x):
	i = 0
	r = b – A @ x
	d = r
	delta_new = r.dot(r)
	delta_0 = delta_new
	while i < i_max and delta_new/delta_0 > epsilon**2:
		q = A @ d	# Matrix-Vector multiplication -> expensive
		alpha = delta_new / d.dot(q)	# dot prod -> expensive
		x = x + alpha*d
		r = b - A @ x # r = r - alpha * q
		delta_old = delta_new
		delta_new = r.dot(r)
		beta = delta_new / delta_old
		d = r + beta * d
		i = i + 1
	return x
```

#### Accelerate the CG Method

- Reduce the time of **sparse-matrix-vector multiplication** of `q = A @ d`: (Piece-wise multiply)
  $$
  A =M +h^{2} \sum_{j=1}^{m} \nabla_{x}^{2} E_{j}(x) \quad \Rightarrow A d=M d+h^{2} \sum_{j=1}^{m} \nabla_{x}^{2} E_{j}(x) d
  $$
  Use `@ti.kernel` -> computing

  (Taichi enables thread local storage automatically for this reduction problem) (Taichi TLS / CUDA Reduction Guide)

- Reduce the **condition number** of $A$:

  The error (Upper bound)
  $$
  \left\|e_{i}\right\|_{A} \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{i}\left\|e_{0}\right\|_{A}
  $$

- Instead of solving $Ax = b$ => solve for $M^{-1} Ax = M^{-1} b$ ($M\sim A$ as much as possible) 

  - Jacobi: $M = \mathrm{diag}(A)$
  - Incomplete Cholesky: $M = \tilde{L}\tilde{L}^{\mathrm{T}}$
  - Multigrid



# Lecture 10  Fluid Simulation 01: The Particle-based (Lagrangian) Methods (21.11.30)

> Lagrangian Method = Particle-Based

## Incompressible Fluid Dynamics

> usually compressible for explosion / shock wave / … ; incompressible for slower dynamics

### Forces for Incompressible Fluids

$$
m\vb{a} = \vb{f} = \vb{f}_{\text{ext}} +\vb f_{\text{pres}} +\vb f_{\text{visc}}
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202093424163.png" alt="image-20211202093424163" style="zoom:50%;" />

### Incompressible Navier–Stokes Equations

The Navier-Stokes Equation ($p = k (\rho_0 - \rho)$) 
$$
\left\{\begin{array}{ll}
\underbrace{\rho \frac{\mathrm{D}v}{\mathrm{D}t}} _{ma/V} = \underbrace{\rho g}_{mg /V} -\underbrace{\grad p}_{\text{adv}} + \underbrace{\mu \laplacian v}_{\text{diff}} & \text{(Momentum Equation)} \\
\div\ v = 0 \Leftrightarrow \frac{\mathrm{D}\rho}{\mathrm{D}t} = \rho (\div\ v ) = 0\ &\text{(Divergence free -- Mass conserving)}
\end{array}\right.
$$

#### The Spatial Derivative Operators

- **Gradient**: $\grad$: $\R^1 \rightarrow \R^3$

  $\grad s  = \left[\frac{\partial s}{\partial x}, \frac{\partial s}{\partial y}, \frac{\partial s}{\partial z} \right]^{\mathrm{T}}$ 

- **Divergence**: $\div$ : $\R^3 \rightarrow \R^1$

  $\div\ v = \frac{\partial v_{x}}{\partial x}+\frac{\partial v_{y}}{\partial y}+\frac{\partial v_{z}}{\partial z}$ 

- **Curl**: $\curl $ : $\R^3 \rightarrow \R^3$ 

  $\curl\ v  = \left[\frac{\partial v_{z}}{\partial y}-\frac{\partial v_{y}}{\partial z}, \frac{\partial v_{z}}{\partial x}-\frac{\partial v_{x}}{\partial z}, \frac{\partial v_{y}}{\partial x}-\frac{\partial v_{x}}{\partial y}\right]^{\mathrm{T}}$ 

- **Laplace**: $\Delta  = \laplacian = \div \grad$: $\R^n \rightarrow \R^n$ (Diffusion)

  $\laplacian s =  \div(\grad s) = \frac{\partial^{2} s}{\partial x^{2}}+\frac{\partial^{2} s}{\partial y^{2}}+\frac{\partial^{2} s}{\partial z^{2}}$ 

## Time Discretization

### Operator Splitting

#### General Steps

Separate the N-S Eqn into 2 parts and use half the timestep to solve each part (**Advection-Projection**)

- Step 1: Input $v_n$, output $v_{n+0.5}$: (Explicit, everything is known)
  - $\rho \frac{\mathrm{D}v}{\mathrm{D}t} = \rho g + \mu \laplacian v$
- Step 2: Input $v_{n+0.5}$, output $v_{n+1}$: (Implicit: $\rho$ and $\grad p$ are unknown)
  - $\rho \frac{\mathrm{D}v}{\mathrm{D}t} = -\grad p$
  - $\div\ v = 0$ 

#### Time Integration

Given $x_n$ & $v_n$ 

- Step 1: **Advection / external and viscosity force** integration
  - Solve: $\mathrm{d}v = g + v\laplacian v_n$
  - Update: $v_{n+0.5} = v_n + \Delta t \ \mathrm{d}v$
- Step 2: **Projection / pressure** solver (Poisson Solver)
  - Solve: $\mathrm{d}v = -\frac{1}{\rho} \grad(k(\rho - \rho_0))$ and $\frac{\mathrm{D} \rho}{\mathrm{D}t} = \div(v_{n+0.5}+\mathrm{d}v) = 0$
  - Update: $v_{n+1} = v_{n+0.5} + \Delta t \ \mathrm{d}v$ 
- Step 3: **Update position**
  - Update: $x _{n+1} = x_n + \Delta t\ v_{n+1}$ 
- Return $x_{n+1}$, $v_{n+1}$ 

### Integration with the Weakly Compressible (WC) Assumption

Storing the density $\rho$ as an individual var that advect with the vel field
$$
\frac{\mathrm D v}{\mathrm D t}=g-\frac{1}{\rho} \nabla p+v \nabla^{2} v\ ;\quad \cancel{\div\ v = 0}\ \text{(weak compressible)}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202105945198.png" alt="image-20211202105945198" style="zoom:67%;" />

#### Time Integration

> Symplectic Euler Integration: In this case step 1 and 2 can be combined since they are both explicit (no order diff)

- Step 1: **Advection / external and viscosity force** integration
  - Solve: $\mathrm{d}v = g + v\laplacian v_n$
  - Update: $v_{n+0.5} = v_n + \Delta t \ \mathrm{d}v$
- Step 2: **Projection / pressure** solver (Use the current-point-evaluated $\rho$)
  - Solve: $\mathrm{d}v = -\frac{1}{\rho} \grad(k(\rho - \rho_0))$ 
  - Update: $v_{n+1} = v_{n+0.5} + \Delta t \ \mathrm{d}v$ 
- Step 3: **Update position**
  - Update: $x _{n+1} = x_n + \Delta t\ v_{n+1}$ 
- Return $x_{n+1}$, $v_{n+1}$ 

## Spatial Discretization (Lagrangian View)

> - Previous knowledge using Lag. View: Mesh-based simulation (FEM)
> - Today: Mesh-free simulation => a simular example: marine balls

### Basic Idea

Cont. view -> **Discrete view** (using particles):
$$
\frac{d v_{i}}{d t}=\underbrace{g-\frac{1}{\rho} \nabla p\left(x_{i}\right)+\nu \nabla^{2} v\left(x_{i}\right)}_{a_i}\ , \text{ where }\nu = \frac{\mu}{\rho_0}
$$
**Time integration** (Symplectic Euler):

$v_i = v_i + \Delta t\ a_i$	&	$x_i = x_i + \Delta t\ v_i$

But still need to evaluate $\rho(x_i)$, $\grad p(x_i)$ & $\laplacian v(x_i)$ 

### Smoothed Particle Hydrodynamics (SPH)

#### Dirac Delta 

##### Trivial Indentity

The Dirac function only tends to infinity at 0 but equals to 0 otherwise. And its overall integral is 1.
$$
f(r) = \int^{\infty}_{-\infty} f(r') \delta (r-r')\ \mathrm{d}r\\
\delta(r)=\left\{\begin{array}{l}
+\infty, \text { if } r=0 \\
0, \text { otherwise }
\end{array} \text { and } \int_{-\infty}^{\infty} \delta(r)\ \mathrm{d} r=1\right.
$$

##### Widen the Dirac Delta

$$
f(r) \approx \int f\left(r^{\prime}\right) W\left(r-r^{\prime}, h\right) d r^{\prime}, \text { where } \lim _{h \rightarrow 0} W(r, h)=\delta(r)
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202114735115.png" alt="image-20211202114735115" style="zoom:50%;" />

$W(r, h)$ - kernel funciton: 

- Symmetric: $W(r, h) = W(-r, h)$
- Sum to unity: $f(r) \approx \int f\left(r^{\prime}\right) W\left(r-r^{\prime}, h\right) d r^{\prime}, \text { where } \lim _{h \rightarrow 0} W(r, h)=\delta(r)$
- Compact support: $W(r, h ) = 0, \text{ if } |r|>h$ (Sampling radius - $h$) 

e.g. $W(r, h) = \left\{\begin{aligned}&\frac{1}{2h},\text{ if } |r|<h\\ &0, \text{otherwise} \end{aligned} \right.$  (Error: $\mathcal{O}(h^2)$, can be reduced by decreasing the sampling radius $h$)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202114217762.png" alt="image-20211202114217762" style="zoom: 67%;" />

##### Finite Probes: Summation

$$
f(r) \approx \int f\left(r^{\prime}\right) W\left(r-r^{\prime}, h\right)\
\mathrm{d} r^{\prime} \approx \sum_{j} V_{j} f\left(r_{j}\right) W\left(r-r_{j}, h\right)
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202114202834.png" alt="image-20211202114202834" style="zoom:67%;" />

##### Smoother Kernel Function

Still use the summation: $f(r) \approx \sum_{j} V_{j} f\left(r_{j}\right) W\left(r-r_{j}, h\right)$. But “trust” the closer probes (the orange & yellow ones) better => Smoother $W$ function

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202115507144.png" alt="image-20211202115507144" style="zoom: 67%;" />

Use a $\C^3$ cont. spline: e.g. (The figure below shows the 1st / 2nd / 3rd order der)
$$
W(r, h)=\sigma_{d} \begin{cases}6\left(q^{3}-q^{2}\right)+1 & \text { for } 0 \leq q \leq \frac{1}{2} \\ 2(1-q)^{3} & \text { for } \frac{1}{2} \leq q \leq 1 \\ 0 & \text { otherwise }\end{cases}\ ;
\quad \text{with } q=\frac{1}{h}\|r\|, \sigma_{1}=\frac{4}{3 h}, \sigma_{2}=\frac{40}{7 \pi h^{2}}, \sigma_{3}=\frac{8}{\pi h^{3}}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202115720769.png" alt="image-20211202115720769" style="zoom:67%;" />

#### Smoothed Particle Hydrodynamics (SPH)

##### Discretization

**1D**:
$$
f(r) \approx \sum_{j} V_{j} f\left(r_{j}\right) W\left(r-r_{j}, h\right)=\sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) W\left(r-r_{j}, h\right)
$$

**2D**:

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202120133759.png" alt="image-20211202120133759" style="zoom:50%;" />

**Par stored in every particles**

- Intrinsic quantities:
  - $h$: support radius
  - $\tilde{h}$: particle radius -> $V$: particle volume
- Time varying quantities:
    - $\rho$: density
    - $v$: velocity
    - $x$: position  


##### Evaluate 2D fields using the SP

$$
f(r_1) 
\approx \frac{m_{2}}{\rho_{2}} f\left(r_{2}\right) W\left(r_{1}-r_{2}, h\right) 
+ \frac{m_{3}}{\rho_{3}} f\left(r_{3}\right) W\left(r_{1}-r_{3}, h\right)
+ \frac{m_{4}}{\rho_{4}} f\left(r_{4}\right) W\left(r_{1}-r_{4}, h\right)
$$

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202124713890.png" alt="image-20211202124713890" style="zoom:50%;" />

##### SPH Spatial Derivatives

The operators will affect only on the kernel func ($W$)

- $\nabla f(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) \nabla W\left(r-r_{j}, h\right)$
- $\nabla\cdot \vb{F}(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} \vb{F}\left(r_{j}\right) \cdot \nabla W\left(r-r_{j}, h\right)$
- $\nabla \times \boldsymbol{F}(r) \approx-\sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) \times \nabla W\left(r-r_{j}, h\right)$
- $\nabla^2 f(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) \nabla^2 W\left(r-r_{j}, h\right)$

##### Improving Approximations for Spatial Derivatives

- $f(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) W\left(r-r_{j}, h\right)$ 
- $\nabla f(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) \nabla W\left(r-r_{j}, h\right)$ 
- Let $f(r)\equiv 1$: (Red line)
  - $1 \approx \sum_{j} \frac{m_{j}}{\rho_{j}} W\left(r-r_{j}, h\right)$		in the graph: not exactly 1.0
  - $0 \approx \sum_{j} \frac{m_{j}}{\rho_{j}} \grad W\left(r-r_{j}, h\right)$    not equal = 0 exactly (even if increase sampling points)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202131525157.png" alt="image-20211202131525157" style="zoom:67%;" />

###### Anti-symmetric Form 

> Better for projection / divergence / …

- => Since $\grad f(r) \equiv \grad f(r) \cdot 1$:

  - $\grad f(r) = \grad f(r) \cdot 1 + f(r) \cdot \grad 1$
  - Or equivalently: $\grad f(r) = \grad f(r) - f(r) \cdot \grad 1$ => the 2 grads can be evaluated using SPH

- $\grad f(r) \approx \sum_{j} \frac{m_{j}}{\rho_{j}} f\left(r_{j}\right) \grad W\left(r-r_{j}, h\right)-f(r) \sum_{j} \frac{m_{j}}{\rho_{j}} \grad W\left(r-r_{j}, h\right) \approx \sum_{j} m_{j} \frac{f\left(r_{j}\right)-f(r)}{\rho_{j}} \nabla W\left(r-r_{j}, h\right)$ (**Anti-symmetric form**, green line)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202134020365.png" alt="image-20211202134020365" style="zoom:67%;" />

###### Symmetric Form

> Better for forces

- A more gen case: 
  - $\grad\left(f(r) \rho^{n}\right)=\grad f(r) \cdot \rho^{n}+f(r) \cdot n \rho^{n-1} \grad\rho$ 
  - $\Rightarrow \grad f(r)=\frac{1}{\rho^{n}}\left(\grad\left(f(r) \cdot \rho^{n}\right)-f(r)\cdot  n \cdot \rho^{n-1} \grad\rho\right)$
- $\grad f(r) \approx \sum_{j} m_{j}\left(\frac{f\left(r_{j}\right) \rho_{j}^{n-1}}{\rho^{n}}-\frac{n f(r)}{\rho}\right) \grad W\left(r-r_{j}, h\right)$ 
- Special Case: when $n=-1$:
  - $\grad f(r) \approx \rho \sum_{j} m_{j}\left(\frac{f\left(r_{j}\right)}{\rho_{j}^{2}}+\frac{f(r)}{\rho^{2}}\right) \grad W\left(r-r_{j}, h\right)$ (**Symmetric form**) 

## Implementation Details (WCSPH)

### Simulation Pipeline

- For i in particles:

  - Search for **neighbors** j -> Apply the support radius $h$ 

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202153003376.png" alt="image-20211202153003376" style="zoom:50%;" /> 

- For i in particles:

  - Sample the $v$ / $\rho$ fields using SPH
    - **Desity**: $\rho_{i}=\sum_{j} \frac{m_{j}}{\rho_{i}} \rho_{j} W\left(r_{i}-r_{j}, h\right)=\sum_{j} m_{j} W_{i j}$  ($W_{ij}$ is for $r_i - r_j$)
    - **Viscosity**: $\nu\grad^{2} v_{i}=v \sum_{j} m_{j} \frac{v_{j}-v_{i}}{\rho_{j}} \grad^{2} W_{i j}$  (Asymmetric form)
    - **Pressure Gradient**: $-\frac{1}{\rho_{i}} \nabla p_{i}=-\frac{\rho_{i}}{\rho_{i}} \sum_{j} m_{j}\left(\frac{p_{j}}{\rho_{j}^{2}}+\frac{p_{i}}{\rho_{i}^{2}}\right) \nabla W_{i j}, \text { where } p=k\left(\rho_{j}-\rho_{0}\right)$ 
  - Compute $f$ / $a$ using **N-S Eqn**
    - $\frac{\mathrm{d} v_{i}}{\mathrm{d} t}=g-\frac{1}{\rho_{i}} \grad p_{i}+v \grad^{2} v_{i}$ 

- For i in particles (Symplectic Euler)

  - Update $v$ using $a$
    - $v_i = v_i +  \Delta t \cdot \frac{\mathrm{d}v_i}{\mathrm{d}t}$
  - Update $x$ using $v$ 
    - $x_i = x_i + \Delta t \cdot v_i$ 

### Boundary Conditions

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202153846215.png" alt="image-20211202153846215" style="zoom:50%;" />

#### Problems of Boundaries

- Insufficient samples 

  - **Free Surface**: lower density and pressure => Generate outward pressure  

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202154016564.png" alt="image-20211202154016564" style="zoom:50%;" />

    -> Sol: Clamp the negative pressure (everywhere), $p = \max (0, k(\rho- \rho_0))$ (only be compressed, cannot be streched)\

  - **Solid Boundary**: lower density and pressure => Fluid leakage (outbound velocity)

    -> Sol: $p = \max (0, k(\rho-\rho_0))$

    for leakage:

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202154114714.png" alt="image-20211202154114714" style="zoom:50%;" /> <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202154350769.png" alt="image-20211202154350769" style="zoom:50%;" /> 
    
    1. **Refect** the outbound veclocity when close to boundary
    2. Pad a layer of **SP underneath** the boundaries: $\rho_{\mathrm{solid}} = \rho_0$ & $v_{\mathrm{solid}} = 0$ => increases numerical viscosity 

### Neighbor Search  

Navie search: $\mathcal{O}(n^2)$ time

=> Use background grid: Common grid size = $h$ (same support radius in SPH)

(Each neighbor search takes 9 grids in 2D and 27 grids in 3D)

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211202154832585.png" alt="image-20211202154832585" style="zoom:50%;" />



# Lecture 11  Fluid Simulation 02: The Grid-based (Eulerian) Methods (21.12.6)

## N-S Equations and Their Time Integration

### Operator Splitting

> A toy example:
>
> Integrate $\frac{\mathrm{d}q}{\mathrm{d}t} = 1 + 2$ (for some quantity $q$ ) => Theoretical sol: $q^{n+1} = q^n + 3\Delta t$
>
> Operator Splitting: $\tilde{q} = q^n + 1\Delta t$ ; $q^{n+1} = \tilde {q} + 2\Delta t$ 

#### Operation Splitting for N-S Eqn

($q$ can be velocity / density / temperature / …)
$$
\frac{\mathrm{D}v}{\mathrm{D}t} = g -\frac{1}{\rho} \grad p + \nu \laplacian v\ ; \quad \div\ v = 0\\
\Rightarrow \left\{\begin{array}{ll}
\text{Advection:}&\frac{\mathrm{D}q} {\mathrm{D}t} = 0\\
\text{Add Forces:}&\frac{\partial v}{\partial t} = g + \nu \laplacian v\\
\text{Projection:}&\frac{\partial v}{\partial t} = -\frac{1}{\rho}\grad p \; \text{ s.t. }\div\ v = 0

\end{array}\right.
$$

#### One Numerical Time-Stepping for N-S Eqn

Given $q^n$:

​	Step 1: Advection: (Solve intuitive formula) => in Lag (trivial), in Euler (not trivial)

​		$q^{n+1} = \text{advect}(v^n, \Delta t, q^n)$  

​		$\tilde v = \text{advect}(v^{n},\Delta t, v^n)$

​	Step 2: Applying forces: (The term of viscosity could be neglected when simulate gas; gravity can be neglected when the gas has similar density as air)

​		$\tilde{\tilde{v}} = \tilde{v}+\Delta t(g+\nu\laplacian \tilde v)$

​	Step 3: Projection: (Solve for pressure and ensure divergence free)

​		$v^{n+1} = \text{project}(\Delta t, \tilde{\tilde{v}})$

​	Return $v^{n+1}$, $q^{n+1}$ 

## From the Lagrangian View to the Eulerian View

In Eulerian grids: Need to record: <u>grid index</u> / velocity / density / temperature / <u>grid size</u> / …

- Lagrangian: Dynamic markers 
  - Pros: Advection (Quantity preservation) / Boundary Condition (Conformal discretization) / Coupling with solids
  - Cons: Spatial derivative / High spatial discretization error / Neigh search / Unbounded distortion / Explicit collision handling
- Eulerian: Static markers
  - (On the opposite side of Lagrangian)

### Spatial Derivatives Using Finite Difference

Spatial derivative: The dimensions can be decoupled when computing the spatial derivatives due to the structural grid  

$\nabla q_{i, j, k}=\left[\begin{array}{l}
\partial q_{i, j, k} / \partial x \\
\partial q_{i, j, k} / \partial y \\
\partial q_{i, j, k} / \partial z
\end{array}\right]$ 

To compute $\partial q/\partial x$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209002832506.png" alt="image-20211209002832506" style="zoom: 67%;" />

- Forward difference: $\left(\frac{\partial q}{\partial x}\right)_i \approx \frac{q_{i+1} - q_i}{\Delta x}$ ;  $\mathcal{O}(\Delta x)$ (Biased)

- Central difference: $\left(\frac{\partial q}{\partial x}\right)_i \approx \frac{q_{i+1} - q_{i-1}}{2\Delta x}$ ;  $\mathcal{O}(\Delta x^2)$ (Unbiased)

  Problem of the central difference: Non-constant functions are able to register a zero spatial derivative  

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209003145314.png" alt="image-20211209003145314" style="zoom: 67%;" />

- Central difference with a “staggered” grid: $\mathcal{O}(\Delta x^2)$ (Unbiased)

  <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209003512741.png" alt="image-20211209003512741" style="zoom: 67%;" />

  Usually store the **velocity** using the staggered fashion (edges); and store the other (scalar) quantities in the grid centers (e.g. temperature / density / pressure)
  $$
  \left(\frac{\partial q}{\partial x}\right)_i \approx \frac{q_{i+1/2} - q_{i-1/2}}{2\Delta x}
  $$

### MAC (Marker-and-Cell) grid

#### Staggered Grid

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209003840801.png" alt="image-20211209003840801" style="zoom:67%;" />
$$
\left\{
\begin{aligned}
v_{i, j}&=\left[\frac{v_{x_{i-1 / 2, j}}+v_{x_{i+1 / 2, j}}}{2}, \frac{v_{y_{i, j-1 / 2}}+v_{y_{i, j+1 / 2}}}{2}\right] \\
v_{i+1 / 2, j}&=\left[v_{x_{i+1 / 2, j}}, \quad \frac{v_{y_{i, j-1 / 2}}+v_{y_{i, j+1 / 2}}+v_{y_{i+1, j-1 / 2}}+v_{y_{i+1, j+1 / 2}}}{4}\right]\\
v_{i, j+1 / 2}&=\left[\frac{v_{y_{i-1 / 2, j}}+v_{y_{i+1 / 2, j}}+v_{y_{i-1 / 2, j+1}}+v_{y_{i+1 / 2, j+1}}}{4}, v_{y_{i, j+1 / 2}}\right]
\end{aligned}
\right.
$$
For a row x col grid (3x3) storage:

- Temperature (or other scalars): 9
- velocity $v_x$ & $v_y$: 12

#### Staggered Grid in Stokes Theorem

Exterior calculus 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209004509483.png" alt="image-20211209004509483" style="zoom:67%;" />

## Advection

### Material Derivative

$$
\frac{\mathrm{D}f}{\mathrm{D}t} = \frac{\partial f}{\partial t} + v\ \cdot \grad f\ ; \text{where } f= f(x,t)\\
\text{Total derivative w.r.t. time: }\ \frac{\mathrm{d}}{\mathrm{d}t} f(x,t) = \frac{\partial f}{\partial x} + \frac{\mathrm{dx}}{\mathrm{d}t}: \frac{\partial f}{\partial x}
= \frac{\partial f}{\partial t} + v\ \cdot \grad f
$$

#### Material Derivative of Vectors  

If $\vb{q}$ is a vector $\vb{q} = [q_x, q_y, q_z]^{\mathrm{T}}$ 
$$
\frac{\mathrm{D} \vb{q}}{\mathrm{D} t}=\frac{\partial \vb{q}}{\partial t}+v \cdot \grad \vb{q} =\frac{\partial\vb q}{\partial t}+\vb{v}:
\left[\begin{aligned}
\left[\begin{array}{l}
\frac{\partial q_{x}}{\partial x} \\
\frac{\partial q_{y}}{\partial x} \\
\frac{\partial q_{z}}{\partial x}
\end{array}\right]  \\
\left[\begin{array}{l}
\frac{\partial q_{x}}{\partial y} \\
\frac{\partial q_{y}}{\partial y} \\
\frac{\partial q_{z}}{\partial y}
\end{array}\right]\\
\left[\begin{array}{l}
\frac{\partial q_{x}}{\partial z} \\
\frac{\partial q_{y}}{\partial z} \\
\frac{\partial q_{z}}{\partial z}
\end{array}\right]
\end{aligned}
\right]=\frac{\partial \vb q}{\partial t}+v_{x}\left[\begin{array}{c}
\frac{\partial q_{x}}{\partial x} \\
\frac{\partial q_{y}}{\partial x} \\
\frac{\partial q_{z}}{\partial x}
\end{array}\right]+v_{y}\left[\begin{array}{c}
\frac{\partial q_{x}}{\partial y} \\
\frac{\partial q_{y}}{\partial y} \\
\frac{\partial q_{z}}{\partial y}
\end{array}\right]+v_{z}\left[\begin{array}{c}
\frac{\partial q_{x}}{\partial z} \\
\frac{\partial q_{y}}{\partial z} \\
\frac{\partial q_{z}}{\partial z}
\end{array}\right]=\frac{\partial \vb q}{\partial t}+\left[\begin{array}{l}
v_{x} \frac{\partial q_{x}}{\partial x}+v_{y} \frac{\partial q_{x}}{\partial y}+v_{z} \frac{\partial q_{x}}{\partial z} \\
v_{x} \frac{\partial q_{y}}{\partial x}+v_{y} \frac{\partial q_{y}}{\partial y}+v_{z} \frac{\partial q_{y}}{\partial z} \\
v_{x} \frac{\partial q_{z}}{\partial x}+v_{y} \frac{\partial q_{z}}{\partial y}+v_{z} \frac{\partial q_{z}}{\partial z}
\end{array}\right] =
\left[\begin{array}{l}
\frac{\partial q_{x}}{\partial t}+v \cdot \nabla q_{x} \\
\frac{\partial q_{y}}{\partial t}+v \cdot \nabla q_{y} \\
\frac{\partial q_{z}}{\partial t}+v \cdot \nabla q_{z}
\end{array}\right]
$$
For velocity (self-advection)
$$
\frac{\mathrm{D} \vb{v}}{\mathrm{D} t}=\frac{\partial \vb{q}}{\partial t}+v \cdot \grad \vb{v} = \left[\begin{array}{l}
\frac{\partial q_{x}}{\partial t}+v \cdot \grad v_{x} \\
\frac{\partial q_{y}}{\partial t}+v \cdot \grad v_{y} \\
\frac{\partial q_{z}}{\partial t}+v \cdot \grad v_{z}
\end{array}\right]
$$

### Quantity Advection

In Eulerian view, quantities flow with the velocity field for $\frac{\mathrm{D} {q}}{\mathrm{D} t}=\frac{\partial {q}}{\partial t}+v \cdot \grad {q} = 0$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209105154349.png" alt="image-20211209105154349" style="zoom:50%;" />

#### Attempt 1: Finite Difference 

$$
\frac{q_{i}^{n+1}-q_{i}^{n}}{\Delta t}+v^{n} \cdot \frac{q_{i+1}^{n}-q_{i-1}^{n}}{2 \Delta x}=0 \Rightarrow q_{i}^{n+1}=q_{i}^{n}-\Delta t v^{n} \cdot \frac{q_{i+1}^{n}-q_{i-1}^{n}}{2 \Delta x}
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209105638323.png" alt="image-20211209105638323" style="zoom:67%;" />

1-D Advection: **Unconditionally Unstable** w/. FTCS

#### Attempt 2: Semi-Lagrangian

$q^{n+1} = q^n$ ? : $ q^{n+1}\left(x^{n+1}\right)=q^{n}\left(x^{n}\right)=q^{n}\left(x^{n+1}-\Delta t v^{n}\right)$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209105843129.png" alt="image-20211209105843129" style="zoom:67%;" />

To find value of $q^n, x^{n+1}-\Delta t v^{n}$: -> **Interpolation**: $q^{n+1}(x^{n+1}) = \text{interpolate}(q^n, x^{n+1}-\Delta t v^{n}) $ 

Usually use Bilinear interpolation in 2D:
$$
q = \mathrm{lerp} (a, b, c,d) = \mathrm{lerp} (\mathrm{lerp} (a, b), \mathrm{lerp} (c,d) )  = \frac{D\cdot a + C\cdot b + B\cdot c + A\cdot D}{A  + B + C + D}
$$

1-D Advection: **Unconditionally Stable** (The peak will stably move forward) => Required $v^n \Delta t < \Delta x$ 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209110616182.png" alt="image-20211209110616182" style="zoom:67%;" />

Assuming: $v^n \Delta t < \Delta x$:  (equiv to a **velocity-aware one-sided finite difference** form)
$$
q_{i}^{n+1}=\frac{\Delta t v^{n}}{\Delta x} q_{i-1}^{n}+\left(1-\frac{\Delta t v^{n}}{\Delta x}\right) q_{i}^{n} = q_i^n - \Delta tv^n \frac{q^n_i - q^n_{i-1}}{\Delta x}
\Rightarrow \frac{q_{i}^{n+1}-q_{i}^{n}}{\Delta t}+v^{n} \cdot \frac{q_{i}^{n}-q_{i-1}^{n}}{2 \Delta x}=0
$$
Problems:

- Increase the numerical dissipation/viscosity

  => Some **better schemes** with less dissipation:

  - Sharper Interpolation (Cubic Hermit spline interpolation)
  - Better error correction schemes
    - MacCormack Method
    - Back and Forth Error Compensation and Correction (BFECC)

- Backtracked “particle” out-of-boundary:

  - Simplest sol: Take the value of the boundary

    <img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209113515073.png" alt="image-20211209113515073" style="zoom:67%;" />


## Projection

### Poisson’s Equation

#### Possion’s Problem

Want to solve $\frac{\partial v}{\partial t} = -\frac{1}{\rho} \grad p$  s.t. $\div\ v = 0$ 

Use finite difference again:
$$
\frac{v_{x_{i-1 / 2, j}}^{n+1}-v_{x_{i-1 / 2, j}}^{n}}{\Delta t}=-\frac{1}{\rho} \frac{p_{i, j}-p_{i-1, j}}{\Delta x}\quad 
\text{s.t. }\underbrace{\frac{v_{x_{i+1 / 2, j}}^{n+1}-v_{x_{i-1 / 2, j}}^{n+1}}{\Delta x}}_\frac{\partial v_x}{\partial x}
+
\underbrace{\frac{v_{i, j+1 / 2}^{n+1}-v_{y_{i, j-1 / 2}}^{n+1}}{\Delta x}}_\frac{\partial v_x}{\partial x}
=0
$$
<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209115951070.png" alt="image-20211209115951070" style="zoom:67%;" />

The condition will not be divergence free => becomes $-\frac{\Delta t}{\rho}\div\grad \ p = -\div \ v^n$ 
$$
-\frac{\Delta t}{\rho}\div\grad \ p = -\div \ v^n\\
\text{or  } \frac{\Delta t}{\rho} \frac{4 p_{i, j}-p_{i+1, j}-p_{i-1, j}-p_{i, j+1}-p_{i, j-1}}{\Delta x^{2}}=-\frac{v_{i+1 / 2, j}^{n}-v_{x_{i-1 / 2, j}}^{n}+v_{y_{i, j-1 / 2}}^{n}-v_{y_{i, j-1 / 2}}^{n}}{\Delta x}
$$
**Another way to achieve Possion’s problem:**

- Want: $\frac{\partial v}{\partial t} = -\frac{1}{\rho} \grad p$  s.t. $\div\ v = 0$ 
- Discretize the pressure equation in time: $v^{n+1} - v^{n} = -\frac{\Delta t}{\rho} \grad p$  s.t. $\div\ v = 0$ 
- Apply divergence operator $\div $ on both sides: $-\div\ v^n =-\frac{\Delta t}{\rho} \div \grad\ p$

#### Pressure Solve

- For every grid: One unknown $p_{i,j}$

- For every grid: One equation: $\frac{\Delta t}{\rho} \frac{4 p_{i, j}-p_{i+1, j}-p_{i-1, j}-p_{i, j+1}-p_{i, j-1}}{\Delta x^{2}}=-\frac{v_{i+1 / 2, j}^{n}-v_{x_{i-1 / 2, j}}^{n}+v_{y_{i, j-1 / 2}}^{n}-v_{y_{i, j-1 / 2}}^{n}}{\Delta x}$

- Require only a linear solver: $Ap =-d$

- All pressure are solved than to update velocity: $v^{n+1}_{x_{i-1/2,j}} = v^{n}_{x_{i-1/2,j}} - \frac{\Delta t}{\rho}\frac{p_{i,j}-p_{i-1,j}}{\Delta x}$ 

  But velocity values are more than pressure values => introducing boundary conditions

### Boundary Conditions  

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209152406791.png" alt="image-20211209152406791" style="zoom:50%;" />

- **Free Surface (Dirichlet)**

  - $p = 0$ for void grids

- **Solid Wall (Neumann)**

  - $v^{n+1}\cdot n= v^{\text{solid}}\cdot n$ or $v^{n+1}_x = v^{\text{solid}}_x,\ v^{n+1}_y = v^{\text{solid}}_y$ 

  - For solid grids:

    $v_{x}^{\text {solid }}=v_{x_{i-1 / 2, j}}^{n+1}=v_{x_{i-1 / 2, j}}^{n}-\frac{\Delta t}{\rho} \frac{p_{i, j}-p_{i-1, j}}{\Delta x} \Rightarrow p_{i-1,j} = p_{i,j} - \frac{\rho\Delta x}{\Delta t }\left(v^n_{x_{i-1/2 ,j}} - v^{\text{solid}}_x \right)$ 

#### Boundary Conditions in Possion’s Problem

- Dirichlet: $p_{i,j+1} = 0$

- Neumann: $p_{i-1,j} = p_{i,j} - \frac{\rho \Delta x}{\Delta t} \left(v^n_{x_{i-1/2,j}}-v^{\text{solid}}_x \right)$ 

- The Possion’s equation with boundaries:
  $$
  \frac{\Delta t}{\rho} \frac{3 p_{i, j}-p_{i+1, j}-p_{i, j-1}}{\Delta x^{2}}=-\frac{v_{x_{i+1 / 2, j}}^{n}-v_{x}^{\text {solid }}+v_{y_{i, j-1 / 2}}^{n}-v_{y_{i, j-1 / 2}}^{n}}{{\Delta x}}
  $$

To solve $Ap= -d$ (linear solvers (see [Lec. 9](https://nikucyan.github.io/sources/Notebooks/Graphics/Taichi_Graphics.html#Linear_Solvers))) 

<img src="https://cdn.jsdelivr.net/gh/Nikucyan/MD_IMG//img/image-20211209153331497.png" alt="image-20211209153331497" style="zoom:67%;" />



