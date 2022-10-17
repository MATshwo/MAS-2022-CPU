#pragma once
SE::SeVec3fSimd* residual = new SE::SeVec3fSimd[numvert];
SE::SeVec3fSimd* residual = new SE::SeVec3fSimd[numvert];
void SeSchwarzPreconditioner::Preconditioning(SeVec3fSimd* z, const SeVec3fSimd* residual, int dim)

void MASPCG((SE::SeSchwarzPreconditioner sh,SeVec3fSimd* z) {

	clock_t time_stt = clock();
	float i = 0;
	LargeVector<glm::vec3> r = (b - A * x);
	LargeVector<glm::vec3> z;
	z.resize(total_points);
	my::MAS(z_temp);
	for (int i = 0; i < total_points; i++)
	{
		z[i] = glm::vec3(z_temp[3 * i + 0], z_temp[3 * i + 1], z_temp[3 * i + 2]);
	}
	LargeVector<glm::vec3> d = z; //p0
	LargeVector<glm::vec3> q;
	float alpha_new = 0;
	float alpha = 0;
	float beta = 0;
	float delta_old = 0;
	float delta_new = dot(r,z); //delta_new = dot(r,z); 初始分子
	float delta0 = delta_new;
	while (i<i_max && delta_new> EPS2 * delta0) {
		q = A * d;
		alpha = delta_new / dot(d, q);
		x = x + alpha * d;
		r = r - alpha * q; //重新将r保存到本地 
		//printf("%d\n", i);
		write_vec(r, b_path);
		my::MAS(z_temp); //再次更新z，这里z_temp是否需要清空为0？无所谓，会直接覆盖！
		for (int i = 0; i < total_points; i++)
		{
			z[i] = glm::vec3(z_temp[3 * i + 0], z_temp[3 * i + 1], z_temp[3 * i + 2]);
		}
		delta_old = delta_new;
		delta_new = dot(z, r);

		beta = delta_new / delta_old; //求出系数β
		d = r + beta * d; //d = z + beta * d; 更新p
		i++;
	}
	//std::cout << "time use in MAS times is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << std::endl;
}


