using UnityEngine;
using System.Collections;


public class Rigid_Bunny : MonoBehaviour 
{
	bool launched 		= false;
	float dt 			= 0.015f;
	Vector3 v 			= new Vector3(0, 0, 0);	// velocity
	Vector3 w 			= new Vector3(0, 0, 0);	// angular velocity
	
	float mass;									// mass
	Matrix4x4 I_ref;							// reference inertia

	/* Def phys par */
	Vector3 g 			= new Vector3(0, -9.8f, 0);		// gravitational accelaration
	Vector3 F;									// gravitational Force (only force)
	// Vector3[] vertices;

	float linear_decay	= 0.999f;				// for velocity decay
	float angular_decay	= 0.98f;				
	float restitution 	= 0.5f;					// for collision (mu_N)
	float mu_T 			= 0.2f;					// tangential restitution


	// Use this for initialization
	void Start () 
	{		
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		F = mass * g;	// Gravitational force F = mg

		float m=1;
		mass=0;
		for (int i=0; i<vertices.Length; i++) 
		{
			mass += m;
			float diag=m*vertices[i].sqrMagnitude;
			I_ref[0, 0]+=diag;
			I_ref[1, 1]+=diag;
			I_ref[2, 2]+=diag;
			I_ref[0, 0]-=m*vertices[i][0]*vertices[i][0];
			I_ref[0, 1]-=m*vertices[i][0]*vertices[i][1];
			I_ref[0, 2]-=m*vertices[i][0]*vertices[i][2];
			I_ref[1, 0]-=m*vertices[i][1]*vertices[i][0];
			I_ref[1, 1]-=m*vertices[i][1]*vertices[i][1];
			I_ref[1, 2]-=m*vertices[i][1]*vertices[i][2];
			I_ref[2, 0]-=m*vertices[i][2]*vertices[i][0];
			I_ref[2, 1]-=m*vertices[i][2]*vertices[i][1];
			I_ref[2, 2]-=m*vertices[i][2]*vertices[i][2];	
		}
		I_ref [3, 3] = 1;
	}
	
	Matrix4x4 Get_Cross_Matrix(Vector3 a)
	{
		//Get the cross product matrix of vector a
		Matrix4x4 A = Matrix4x4.zero;
		A [0, 0] = 0; 
		A [0, 1] = -a [2]; 
		A [0, 2] = a [1]; 
		A [1, 0] = a [2]; 
		A [1, 1] = 0; 
		A [1, 2] = -a [0]; 
		A [2, 0] = -a [1]; 
		A [2, 1] = a [0]; 
		A [2, 2] = 0; 
		A [3, 3] = 1;
		return A;
	}

	Matrix4x4 Add(Matrix4x4 A, Matrix4x4 B) 
	{
		// Get the sum of matrices A and B
		Matrix4x4 C = Matrix4x4.zero;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				C[i, j] = A[i, j] + B[i, j];
			}
		}
		return C;
	}

	Matrix4x4 Scalar_Prod(Matrix4x4 A, float a)
	{
		// Get the component-wise product of a matrix with a scalar
		Matrix4x4 B = Matrix4x4.zero;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++) {
				B[i, j] = A[i, j] * a;
			}
		}
		return B;
	}

	Quaternion Quaternion_Add(Quaternion a, Quaternion b) {
		Quaternion c = new Quaternion(0.0f, 0.0f, 0.0f, 0);
		c.x = b.x + a.x;
		c.y = b.y + a.y;
		c.z = b.z + a.z;
		return c;
	}

	// In this function, update v and w by the impulse due to the collision with
	//a plane <P, N>
	void Collision_Impulse(Vector3 P, Vector3 N)
	{
		Matrix4x4 R0 = Matrix4x4.Rotate(transform.rotation);
		Vector3 x = transform.position;
		Mesh mesh = GetComponent<MeshFilter>().mesh;
		Vector3[] vertices = mesh.vertices;

		Vector3 v_avg = new Vector3 (0.0f, 0.0f, 0.0f);
		Vector3 Rr_avg = new Vector3 (0.0f, 0.0f, 0.0f);
		int n = 0;	// to count the amount of contact point at the same timestep


		for (var i = 1; i < vertices.Length; i++) {

			// Collision Detection	
			Vector3 Rr_i = R0 * vertices[i];	
			Vector3 x_i = x + Rr_i;	// x_i = x + Rr_i 
			float phi = Vector3.Dot((x_i - P), N);	// phi = (x - P) * N
			

			if (phi < 0) {	// Collision occurs

				Vector3 w_i = Get_Cross_Matrix(w) * Rr_i;
				Vector3 v_i = v + w_i;	// v_i = v + w x Rr_i
				if (Vector3.Dot(v_i, N) < 0) {	// need to compute impulse
					
					// storage the sum of v_i & Rr_i for various contact points
					v_avg +=  v_i;
					Rr_avg += Rr_i;
					n++; // contact times
				}
			}			
		}
		if (n > 0) {	// at least one contact points

			// compute the actual average velocity_i and Rr_i
			v_avg = v_avg / n;
			Rr_avg = Rr_avg / n;

			// velocity before collision
			Vector3 v_N = Vector3.Dot(v_avg, N) * N;
			Vector3 v_T = v_avg - v_N;
			// new velocity
			Vector3 vnew_N = - restitution * v_N;
			float a = Mathf.Max(1 - (mu_T * (1 + restitution) * v_N.magnitude / v_T.magnitude), 0);
			Vector3 vnew_T = a * v_T;
			Vector3 vnew = vnew_T + vnew_N;	 

			// solve for impulse j
			Matrix4x4 I = Matrix4x4.identity;
			Matrix4x4 Rrc_i = Get_Cross_Matrix(Rr_avg);	// converted Rr_i
					
			Matrix4x4 K = Add(Scalar_Prod(I, (1/mass)), Scalar_Prod(Rrc_i * I_ref.inverse * Rrc_i, -1));
			Vector3 j = K.inverse * (vnew - v_avg);

			// update v and w
			v = v + j * (1/mass); // v = v + j / mass
			Vector3 dw = I_ref.inverse * (Rrc_i * j);
			w = w + dw;

			// restitution *= 0.5f;
			// mu_T *= 0.2f; // not sure if should be updated every loop
		}	
	}

	// Update is called once per frame
	void Update () 
	{
		//Game Control
		if(Input.GetKey("r"))
		{
			transform.position = new Vector3 (0, 0.6f, 0);
			transform.rotation = new Quaternion (0, 0, 0, 0);
			restitution = 0.5f;
			launched=false;
		}
		if(Input.GetKey("l"))
		{
			v = new Vector3 (5, 2, 0);
			w = new Vector3 (5, 5, 0); // given some init vel
			launched=true;
		}

		if (launched == false) {
			v = new Vector3 (0, 0, 0);
			w = new Vector3 (0, 0, 0);
		}
		else
		{
			// Part I: Update velocities	
			v += dt/2 * g;	// v_[0.5] = v + dt/2 * f/M
			//w += dt/2 * I.inverse * tau;	// w_[0.5] = w[0] + dt/2 * I.inverse * tau[0]

			v *= linear_decay;	w *= angular_decay;	// damping
				
			// Part II: Collision Impulse
			Collision_Impulse(new Vector3(0, 0.01f, 0), new Vector3(0, 1, 0));
			Collision_Impulse(new Vector3(2, 0, 0), new Vector3(-1, 0, 0));

			// Part III: Update position & orientation
			//Update linear status
			Vector3 x    = transform.position;
			x += dt * v;
			//Update angular status
			Quaternion q = transform.rotation;
			q = Quaternion_Add(q, new Quaternion(dt/2 * w.x, dt/2 * w.y, dt/2 * w.z, 0) * q);

			// Part IV: Assign to the object
			transform.position = x;
			transform.rotation = q;

			v += dt/2 * g;	//w += dt/2 * I_ref.inverse;	// leapfrog integrator -> add another dt/2 
		
		}
	}
}
