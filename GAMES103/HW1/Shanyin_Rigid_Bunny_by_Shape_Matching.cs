using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rigid_Bunny_by_Shape_Matching : MonoBehaviour
{
    public bool launched = false;
    Vector3[] X;    // position for every vertex
    Vector3[] Q;    // r
    Vector3[] V;    // velocity
    Matrix4x4 QQt = Matrix4x4.zero;
	// Vector3[] Y;    // update position

    // set the 2 walls' P and N the same as the prev script
    Vector3[] P = {new Vector3(0, 0.01f, 0), new Vector3(2, 0, 0)};
    Vector3[] N = {new Vector3(0, 1, 0), new Vector3(-1, 0, 0)};

	Vector3 g = new Vector3(0, -9.8f, 0);

	float linear_decay	= 0.999f;				// for velocity decay
	//float angular_decay	= 0.98f;
	float restitution 	= 0.5f;					// for collision (mu_N)
	float mu_T 			= 0.2f;					// tangential restitution



    // Start is called before the first frame update
    void Start()
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        V = new Vector3[mesh.vertices.Length];
        X = mesh.vertices;
        Q = mesh.vertices;
        

        //Centerizing Q.
        Vector3 c = Vector3.zero;
        for (int i = 0; i < Q.Length; i++)
            c += Q[i];
        c /= Q.Length;
        for (int i = 0; i < Q.Length; i++)
            Q[i] -= c;

        //Get QQ^t ready.
        for (int i = 0; i < Q.Length; i++)
        {
            QQt[0, 0] += Q[i][0] * Q[i][0];
            QQt[0, 1] += Q[i][0] * Q[i][1];
            QQt[0, 2] += Q[i][0] * Q[i][2];
            QQt[1, 0] += Q[i][1] * Q[i][0];
            QQt[1, 1] += Q[i][1] * Q[i][1];
            QQt[1, 2] += Q[i][1] * Q[i][2];
            QQt[2, 0] += Q[i][2] * Q[i][0];
            QQt[2, 1] += Q[i][2] * Q[i][1];
            QQt[2, 2] += Q[i][2] * Q[i][2];
        }
        QQt[3, 3] = 1;

        for (int i = 0; i < X.Length; i++)
            V[i][0] = 4.0f;

        Update_Mesh(transform.position, Matrix4x4.Rotate(transform.rotation), 0);
        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;
    }

    // Polar Decomposition that returns the rotation from F.
    Matrix4x4 Get_Rotation(Matrix4x4 F)
    {
        Matrix4x4 C = Matrix4x4.zero;
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    C[ii, jj] += F[kk, ii] * F[kk, jj];

        Matrix4x4 C2 = Matrix4x4.zero;
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    C2[ii, jj] += C[ii, kk] * C[jj, kk];

        float det = F[0, 0] * F[1, 1] * F[2, 2] +
                        F[0, 1] * F[1, 2] * F[2, 0] +
                        F[1, 0] * F[2, 1] * F[0, 2] -
                        F[0, 2] * F[1, 1] * F[2, 0] -
                        F[0, 1] * F[1, 0] * F[2, 2] -
                        F[0, 0] * F[1, 2] * F[2, 1];

        float I_c = C[0, 0] + C[1, 1] + C[2, 2];
        float I_c2 = I_c * I_c;
        float II_c = 0.5f * (I_c2 - C2[0, 0] - C2[1, 1] - C2[2, 2]);
        float III_c = det * det;
        float k = I_c2 - 3 * II_c;

        Matrix4x4 inv_U = Matrix4x4.zero;
        if (k < 1e-10f)
        {
            float inv_lambda = 1 / Mathf.Sqrt(I_c / 3);
            inv_U[0, 0] = inv_lambda;
            inv_U[1, 1] = inv_lambda;
            inv_U[2, 2] = inv_lambda;
        }
        else
        {
            float l = I_c * (I_c * I_c - 4.5f * II_c) + 13.5f * III_c;
            float k_root = Mathf.Sqrt(k);
            float value = l / (k * k_root);
            if (value < -1.0f) value = -1.0f;
            if (value > 1.0f) value = 1.0f;
            float phi = Mathf.Acos(value);
            float lambda2 = (I_c + 2 * k_root * Mathf.Cos(phi / 3)) / 3.0f;
            float lambda = Mathf.Sqrt(lambda2);

            float III_u = Mathf.Sqrt(III_c);
            if (det < 0) III_u = -III_u;
            float I_u = lambda + Mathf.Sqrt(-lambda2 + I_c + 2 * III_u / lambda);
            float II_u = (I_u * I_u - I_c) * 0.5f;


            float inv_rate, factor;
            inv_rate = 1 / (I_u * II_u - III_u);
            factor = I_u * III_u * inv_rate;

            Matrix4x4 U = Matrix4x4.zero;
            U[0, 0] = factor;
            U[1, 1] = factor;
            U[2, 2] = factor;

            factor = (I_u * I_u - II_u) * inv_rate;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    U[i, j] += factor * C[i, j] - inv_rate * C2[i, j];

            inv_rate = 1 / III_u;
            factor = II_u * inv_rate;
            inv_U[0, 0] = factor;
            inv_U[1, 1] = factor;
            inv_U[2, 2] = factor;

            factor = -I_u * inv_rate;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    inv_U[i, j] += factor * U[i, j] + inv_rate * C[i, j];
        }

        Matrix4x4 R = Matrix4x4.zero;
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                for (int kk = 0; kk < 3; kk++)
                    R[ii, jj] += F[ii, kk] * inv_U[kk, jj];
        R[3, 3] = 1;
        return R;
    }

    // Update the mesh vertices according to translation c and rotation R.
    // It also updates the velocity.
    void Update_Mesh(Vector3 c, Matrix4x4 R, float inv_dt)
    {
        for (int i = 0; i < Q.Length; i++)
        {
            Vector3 x = (Vector3)(R * Q[i]) + c;

            V[i] += (x - X[i]) * inv_dt;
            X[i] = x;
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = X;
    }

    void Collision(float inv_dt)
    {
        // loop for every wall
		for (int i = 0; i < 2; i++) {
            for (int j = 0; j < V.Length; j++) {
                Vector3 x_j = X[j];
                Vector3 p_i = P[i];
                // Vector3 n_i = N
                // detection
			    float phi = Vector3.Dot((x_j - p_i), N[i]);	// phi = (x - P) * N

			    if (phi < 0) {	// Collision occurs
                    if (Vector3.Dot(V[j], N[i]) < 0) {  // need to compute  
                        // before collision
                        Vector3 v_N = Vector3.Dot(V[j], N[i]) * N[i];
                        Vector3 v_T = V[j] - v_N;
                        // new velocity
                        Vector3 vnew_N = -restitution * v_N;
                        float a = Mathf.Max(1 - (mu_T * (1 + restitution) * v_N.magnitude / v_T.magnitude), 0);
			            Vector3 vnew_T = a * v_T;
                        // v_new is directly new V[j]
			            V[j] = vnew_T + vnew_N;	 
                        X[j] += X[j] + V[j] * inv_dt;
                    }
                }
            }  
		}
    }

    // Update is called once per frame
    void Update()
    {
        // Get the keyboard control
		if (Input.GetKey("r"))
        {
            transform.position = Vector3.zero;
        	transform.rotation = Quaternion.identity;
            launched = false;
        }
        if (Input.GetKey("l"))
        {
            launched = true;
        }

        float dt = 0.015f;

        // main loop
        if (launched == true)
        {
            //Step 1: run a simple particle system.
            for (int i = 0; i < V.Length; i++)
            {
				V[i] += g * dt;     // v_i = v_i + dt * f/m (= dt * g)
				X[i] = X[i] + V[i] * dt;    // y_i = x_i + v_i * dt 

				V[i] *= linear_decay;
            }

            //Step 2: Perform simple particle collision.
            Collision(1 / dt);


            // Step 3: Use shape matching to get new translation c and 
            // new rotation R. Update the mesh by c and R.
            //Shape Matching (translation)
            Vector3 c = new Vector3 (0, 0, 0);
            for (int i = 0; i < V.Length; i++) {
                c += X[i];
            }
            c = c / V.Length;   // c = sum{y_i} / N

            //Shape Matching (rotation)
            // compute Matrix A
            // A = (sum{(y_i - c) * r_i^T})(sum{r_i r_i^T})^{-1}
            Matrix4x4 A = Matrix4x4.zero;
            //Matrix4x4 QQt_i = QQt.inverse;
            for (int i = 0; i < Q.Length; i++) {
                Vector3 X_c = X[i] - c;
                for (int j = 0; j < 3; j++) 
                    for (int k = 0; k < 3; k++) 
                        A[j, k] += X_c[j] * Q[i][k];
                A[3, 3] = 1; 
            }

            A *= QQt.inverse;   // QQt.inverse = sum(r_i r_i^T)^{-1}

            // update R = Polar(A)
            Matrix4x4 R = Get_Rotation(A);


            Update_Mesh(c, R, 1/dt);
        }


    }
}
