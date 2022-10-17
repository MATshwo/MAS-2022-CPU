#include<iostream>
#include"SeSchwarzPreconditioner.h"
#include<emmintrin.h>

#include "edge_face.h"
#include "read_sysAb.h"


int main()
{
	const int max_interation = 2;
	const float eple = 0.000000001f;
	//Loading data
	//1��AllocatePrecoditioner input
	const int row_num = 21;
	const int col_num = 21;
	const int numvert = row_num*col_num; // ԭʼȡ������=8,8*8
	const int numedge = 1240;  //7*7*3+��8-1��*2=161   �����һ��&���һ�У����ඥ����Ե��������ߣ����һ��/�е�����/�ж�����-1����
	const int numface = 800; //7*7*2=98 �����һ��&���һ�У����ඥ����Ե�������������
	//char A_path[] = "./Hessian_matrix1.txt";
	//char b_path[] = "./Residual1.txt";
	char A_path[] = "./first_a.txt";
	char b_path[] = "./first_b.txt";
	//��ʼ����������
	SE::SeVec3fSimd* m_positions = new SE::SeVec3fSimd[numvert*1.5];
	int vcount = 0;
	for (int i = 0; i < row_num; i++)
	{
		for (int j = 0; j < col_num; j++)
		{
			SE::SeVec3fSimd temp((float)i/4, (float)j / 4, 0, 0); //����Ϊ5
			m_positions[vcount] = temp;
			vcount++;
		}
	}
	//�������ʼ��(e1,e2,0,0),����=m_numedge
	SE::Int4* m_edges = new SE::Int4[numedge * 1.5];
	my::gen_edge(numvert,row_num,col_num ,m_edges, m_positions);
	//�������ʼ��(x1,x2,x3,0),����=m_numface
	SE::Int4* m_faces = new SE::Int4[numface * 1.5];
	my::gen_face(numvert,row_num,col_num, m_faces,m_positions);

	//neighbor��Ϣ
	int start_temp[numvert + 1];
	int id_length = my::neibor_start(numvert,row_num,col_num,start_temp); // last value in start_temp
	int *idx_temp = new int[id_length * 1.5];
	my::neibor_idx(numvert,row_num,col_num, idx_temp);
	std::vector<int> start(start_temp,start_temp+numvert+1); 
	std::vector<int> idx(idx_temp, idx_temp+id_length);
	start.shrink_to_fit();
	idx.shrink_to_fit();
	std::vector<int> values(id_length, 1);  
	SE::SeCsr<int> m_neighbours_temp(start, idx, values); 
	SE::SeCsr<int>* m_neighbours = &m_neighbours_temp; 

	//2��PreparePreconditioner input
	//���Խ����ϵ�3*3����
	//ֱ�Ӵ���֪ģ���ж�ȡ
	float* sys_A = new float[3 * 3 * numvert];
	float* sys_b = new float[3 * numvert];
	sys_A = my::read_data(A_path, 3 * 3 * numvert);
	sys_b = my::read_data(b_path, 3 * numvert);
	//std::cout << sys_b[3] << std::endl; //testing code!

	SE::SeMatrix3f* diagonal = new SE::SeMatrix3f[numvert * 1.5];
	for (int a = 0; a < numvert; a++)
	{
		//float r[9] = { sys_A[9*a],sys_A[9*a+1],sys_A[9*a+2],sys_A[9*a+3],sys_A[9*a+4],sys_A[9*a+5],sys_A[9*a+6],sys_A[9*a+7],sys_A[9*a+8] };
		//SE::SeMatrix3f nows = SE::SeMatrix3f::Identity(1);
		float r[9] = { 1,0,1,0,1,0,1,0,0 }; //A��Ҫ�ǶԳ�����
		//if (a > 31)
		//{
		//	SE::SeMatrix3f nows = SE::SeMatrix3f::Identity(2); 
		//	diagonal[a] = nows;
		//	continue;
		//}
		SE::SeMatrix3f nows(r);
		diagonal[a] = nows;
	}
	//csrrange����start,�洢ÿ���ʵ��׸�offdiagnoal���б�csrrange[]�е�index  
	int* csrRanges = new int[(numvert + 1) * 1.5]; //��ʾ��i�������ǰi-1������һ���ж����ھӣ�����˵Ӧ��=start
	int nn = 0;
	for (auto i = start.begin(); i != start.end(); i++) { csrRanges[nn] = (int)*i; nn++; }
	
	//�ǶԽ��ߵ�3*3����
	SE::SeMatrix3f* csrOffDiagonals = new SE::SeMatrix3f[id_length * 1.5];
	for (int a = 0; a < id_length; a++)
	{
		SE::SeMatrix3f nows = SE::SeMatrix3f::Identity(0); //�ǶԽ�Ԫ��ָ����ͬ���½���Ĳ�ͬ
		csrOffDiagonals[a] = nows;
	}
	
	//�ο�����PSCC
	const SE::EfSet* efSets = {}; //�Ȳ�����ef��ײ
	const SE::EeSet* eeSets = {};
	const SE::VfSet* vfSets = {};
	unsigned int* efCounts = new unsigned int[numedge + 1]{ 0 }; //�����б��ʼ��Ϊ0
	unsigned int* eeCounts = new unsigned int[numedge + 1]{ 0 };
	unsigned int* vfCounts = new unsigned int[numvert + 1]{ 0 };
	
	//unsigned int* efCounts = new unsigned int[numedge + 1]{ 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 }; //�����б��ʼ��Ϊ0
	//unsigned int* eeCounts = new unsigned int[numedge + 1]{ 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	//unsigned int* vfCounts = new unsigned int[numvert + 1]{ 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	
	//3��Preconditioning input
	//std::vector<SE::SeVec3fSimd> z;
	SE::SeVec3fSimd* z = new SE::SeVec3fSimd[numvert * 1.5];
	//SE::SeVec3fSimd* z = new SE::SeVec3fSimd[2*numvert]; //���µĲ���
	//residual �Ҳೣ����
	SE::SeVec3fSimd* residual = new SE::SeVec3fSimd[numvert * 1.5];
	for (int a = 0; a < numvert; a++)
	{
		//float temp[3] = { 0.1,0.2,0.3 };
		//SE::SeVec3fSimd nows(sys_b[a*3+0],sys_b[3*a+1],sys_b[3*a+2]); //��Ϊ����level����
		SE::SeVec3fSimd nows(0.9,1.2,0.3);
		residual[a] = nows;
	}
	int dim = 1; //un-useful
	clock_t time_stt = clock();
	//4��prepare preconditioner 
	SE::SeSchwarzPreconditioner sh;
	sh.m_positions = m_positions;
	sh.m_edges = m_edges;
	sh.m_faces = m_faces;
	sh.m_neighbours = m_neighbours; //��neighborΪ��ͨ����,�˴���ֵ���&
	
	sh.AllocatePrecoditioner(numvert, numedge, numface);
	std::cout << "AllocatePrecoditioner run success!" << std::endl;
	
	//PCG����ǰ���ã�����A�������
	//��Ҫ��ʱ���裡����Ԥ������M
	sh.PreparePreconditioner(diagonal, csrOffDiagonals, csrRanges, efSets, eeSets, vfSets, efCounts, eeCounts, vfCounts);
	std::cout << "PreparePreconditioner run success!" << std::endl;
	//PCG����,�õ�ÿ�θ��µĲ�����
	
	
	// CALL PCG iteration_ PCG(z,r)
	//for (int i = 0; i < max_interation; i++)
	//{	
	//	sh.Preconditioning(z, residual, dim); 
	//}

	sh.Preconditioning(z, residual, dim);
	std::cout << "Preconditioning run success!" << std::endl;
	// 40�� 42ms
	std::cout << "time use in MAS solver * 40 times is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << std::endl;
	for (int k = 0; k < 10; k++)
	{
		//show result
		unsigned int kk = sh.m_MapperSortedGetOriginal[k];
		//printf(" %6f  %6f  %6f\n", z[kk].x, z[kk].y, z[kk].z); //z��ԭʼ�ģ�����Ҫת���ˣ�
		printf(" %6f  %6f  %6f\n", z[k].x, z[k].y, z[k].z);
	}
	delete m_positions;
	delete m_edges;
	delete m_faces;
	delete diagonal;
	delete csrOffDiagonals;
	delete csrRanges;
	delete idx_temp;
	delete z;
	delete residual;
	return 0;
}
