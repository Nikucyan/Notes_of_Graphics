# Homework Assignments of GAMES103

> **GAMES103**: 
>
> [Repo](https://github.com/Nikucyan/Notes_of_Graphics/tree/main/GAMES103) | [Notebook](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES103.html) | [Homeworks](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES103_Homework) | [Course Site](http://games-cn.org/games103/) | [Videos](https://www.bilibili.com/video/BV12Q4y1S73g) 

[![](https://img.shields.io/badge/HW-Codes-blue)](https://github.com/Nikucyan/Notes_of_Graphics/tree/main/GAMES103/Homework_Assignments/)



## Lab 1: Angry Bunny

**Rigid Body Collision**

Given: Unity package `bunny.unitypackage` and prepared imcompleted scripts.

### Tasks

#### 1. Basic Tasks: (Impulse method)

1. Position Update in `Update` function: 

   Update both the **position** & the **orientation** by **Leapfrog integration**.

   Disable the motion and position update if `launch = false`.

2. Velocity Update:

   Calculate the **gravity** force to update the velocity.

   Produce **damping effects** by multiply the velocity by decay factors.

   No need to calculate torque or angular velocity.

3. Collision Detection in `Collision_Impulse` function:

   Calculate the position and velocity of **every mesh vertex** and determine if in collision.

   If collide, compute the **average** of all colliding vertices.
   
3. Collision Response:

   Apply the impulse-based method to calculate the **impulse** `j` for the average colliding position.

   Update the **linear and angular velocities** accordingly.
   
3. Collision with the Wall:

   Collide with the side wall as well.

#### 2. Bonus Tasks: (Shape matching)

Complete the other script with shape matching method.

### Results

1. Impulse Method:

   ![Impulse](https://cdn.jsdelivr.net/gh/Nikucyan/Notes_of_Graphics/GAMES103/Homework_Assignments/HW1/IMG_1710.GIF)

2. Shape Matching:

   ![Shape_Matching](https://cdn.jsdelivr.net/gh/Nikucyan/Notes_of_Graphics/GAMES103/Homework_Assignments/HW1/IMG_1709.GIF)



## Lab 2: Cloth Simulation

**Soft-matter Simulation**

Given: `cloth.unitypackage ` (a retangular cloth piece and a sphere). The cloth piece was defaultly set with 21 x 21 =  441 vertices

Requirement: drag the sphere and have the cloth interactive simulation with the sphere

### Tasks [WIP]

#### 1. Implicit Cloth Solver

1. Initial Setup: 

   For every vertex, apply damping to velocity and/or other initial conditions

2. Gradient Calculation: 

   Compute the gradient `G` by applying both the **gravitaitonal force** and the **spring force**

3. Finishing: (Right after the gradient computation step, by **Jacobi**) 

   Instead of using Newton’s method to solve the nonlinear optimization problem, here the Hessian matrix is considered as a **diagonal** matrix

   The vertices are updated 32 times to ensure the quality

   Finally, calculate the **velocity** and **assign** `X` to `mesh.vertices` 

4. Chebyshev Acceleration: 

   Jacobi with **Chebyshev** semi-iterative method also applies to nonlinear optimization

5. Sphere Collision: In the `Collision_Handling` function

   Calculate the distance from every vertex to the sphere center to detect


#### 2. Position-Based Dynamics (PBD)

1. Initial Setup: In `Update` function)Set the PBD solver as a particle system.

   For every vertex, damp the **velocity** and update by gravity. And also update the **position** `x`.

2. Strain Limiting: In `Strain_Limiting` function, in **Jacobi** fashion

   Define 2 **temp arrays** `sum_x[]` and `sum_n[]` to store the sums of vertex **position** updates and vertex **count** updates (Set as 0 initially)

   For every edge `e` connecting `i` and `j` update the sum arrays

   Update the vertex velocity

   (Should be done multiple times)

3. Sphere Collision (Same as implicit method)   

### Results

1. Implicit Method: [Using [WebGL](https://nikucyan.github.io/sources/Notebooks/Graphics/GAMES103_HW2/Implicit.html)]

   ![Implicit](https://github.com/Nikucyan/Notes_of_Graphics/blob/main/GAMES103/Homework_Assignments/HW2/Implicit_new.GIF?raw=true)

2. PBD:



   

   

​      
